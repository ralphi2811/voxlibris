"""Ouvrier : exécute les tâches déposées par l'interface web.

Processus distinct de l'interface, pour deux raisons. La première est évidente : une
synthèse dure des dizaines de minutes et ne peut pas se dérouler dans une requête HTTP.
La seconde l'est moins et pèse plus lourd : seul cet ouvrier a besoin de CUDA et de la
pile PyTorch, ce qui laisse l'image de l'interface à quelques centaines de mégaoctets là
où celle-ci en pèse huit gigaoctets.

    python -m voxlibris.worker
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path

from .assemble import build
from .normalize import build_segments
from .project import Project
from .web.jobs import Job, Queue, default_queue, workspace

logger = logging.getLogger("voxlibris.worker")
IDLE_SLEEP = 1.0

_stop = False


def _request_stop(*_) -> None:
    global _stop
    _stop = True
    logger.info("arrêt demandé : la tâche en cours va s'achever avant de rendre la main")


def run_normalize(project: Project, job: Job, queue: Queue) -> None:
    counts = build_segments(project.text_dir, project.segments_dir)
    queue.report(job.id, 1.0, f"{sum(counts.values())} segments écrits")


def run_synth(project: Project, job: Job, queue: Queue) -> None:
    from .tts.backends import load
    from .tts.synth import load_segments, profile_for_voice, synthesize_chapter

    backend = job.params.get("backend", "xtts")
    voice = job.params.get("voice")
    force = bool(job.params.get("force"))
    device = job.params.get("device", "cuda")

    paths = sorted(project.segments_dir.glob("ch*.jsonl"))
    if not paths:
        raise RuntimeError("Aucun segment : lancer la normalisation d'abord.")

    queue.report(job.id, 0.0, f"chargement du moteur {backend}")
    engine = load(backend, voice, device)
    chosen = getattr(engine, "voice", backend)

    signature = f"{backend}/{chosen}"
    changed = project.voice is not None and f"{project.backend}/{project.voice}" != signature
    if changed:
        queue.report(job.id, message="voix différente de la précédente : tout est resynthétisé")
    project.backend, project.voice = backend, chosen
    project.save()

    profile = profile_for_voice(engine, load_segments(paths[0]), project.calibration_file)
    queue.report(job.id, message=f"débit calibré à {profile.chars_per_second} car/s")

    warnings: list[str] = []
    for index, path in enumerate(paths):
        target = project.wav_dir / f"{path.stem}.wav"
        if target.exists() and not (force or changed):
            queue.report(job.id, (index + 1) / len(paths), f"{path.stem} déjà synthétisé")
            continue
        segments = load_segments(path)
        queue.report(
            job.id,
            index / len(paths),
            f"{path.stem} — {segments[0].title} ({len(segments)} segments)",
        )
        result = synthesize_chapter(engine, segments, profile)
        result.write(target)
        warnings += result.warnings
        queue.report(
            job.id, (index + 1) / len(paths), f"{path.stem} → {result.duration / 60:.1f} min"
        )

    if warnings:
        queue.report(job.id, message=f"{len(warnings)} segment(s) à vérifier :")
        for warning in warnings:
            queue.report(job.id, message=f"  {warning}")
    else:
        queue.report(job.id, message="contrôle qualité : aucun segment hors tolérance")


def run_proofread(project: Project, job: Job, queue: Queue) -> None:
    """Interroge le modèle de langage sur les formes suspectes.

    Le résultat est déposé sur disque, et rien d'autre : c'est l'éditeur de relecture qui
    présentera les propositions, une par une, à l'arbitrage d'un humain.
    """
    from .proofread import suggest

    texts = project.chapter_texts()
    queue.report(job.id, 0.1, f"{len(texts)} chapitre(s) à examiner")
    report = suggest(texts, language=project.language)
    project.suggestions_file.parent.mkdir(parents=True, exist_ok=True)
    project.suggestions_file.write_text(report.to_json(), encoding="utf-8")

    if report.error:
        raise RuntimeError(report.error)
    queue.report(
        job.id,
        1.0,
        f"{len(report.suggestions)} proposition(s) sur {report.asked} forme(s) soumises, "
        f"{len(report.rejected)} écartée(s) par les garde-fous. Rien n'a été modifié.",
    )


def run_sample(project: Project, job: Job, queue: Queue) -> None:
    """Synthétise un même extrait avec plusieurs voix, pour choisir à l'oreille."""
    from .tts.backends import load
    from .tts.quality import QualityProfile
    from .tts.synth import load_segments, synthesize_chapter

    choices: list[dict] = job.params.get("voices", [])
    count = int(job.params.get("segments", 8))
    paths = sorted(project.segments_dir.glob("ch*.jsonl"))
    if not paths:
        raise RuntimeError("Aucun segment : lancer la normalisation d'abord.")
    segments = load_segments(paths[0])[:count]

    target_dir = project.out_dir / "samples"
    target_dir.mkdir(parents=True, exist_ok=True)

    for index, choice in enumerate(choices):
        backend, voice = choice["backend"], choice.get("voice")
        queue.report(job.id, index / len(choices), f"{backend} / {voice}")
        try:
            engine = load(backend, voice, job.params.get("device", "cuda"))
            result = synthesize_chapter(engine, segments, QualityProfile())
            name = f"{backend}--{(voice or 'defaut').replace(' ', '_')}.wav"
            result.write(target_dir / name)
            queue.report(job.id, message=f"  → {name}")
        except Exception as error:  # une voix indisponible n'arrête pas la comparaison
            queue.report(job.id, message=f"  échec {backend}/{voice} : {error}")
    queue.report(job.id, 1.0, "échantillons prêts")


def run_assemble(project: Project, job: Job, queue: Queue) -> None:
    cover = Path(project.source) if project.source else None
    m4b = build(
        wav_dir=project.wav_dir,
        segments_dir=project.segments_dir,
        out_dir=project.out_dir,
        meta=project.metadata,
        cover_source=cover,
        write_mp3s=not job.params.get("skip_mp3"),
        on_progress=lambda line: queue.report(job.id, message=line),
    )
    queue.report(job.id, 1.0, f"{m4b.name} prêt")


HANDLERS = {
    "normalize": run_normalize,
    "proofread": run_proofread,
    "synth": run_synth,
    "sample": run_sample,
    "assemble": run_assemble,
}


def execute(job: Job, queue: Queue, root: Path) -> None:
    handler = HANDLERS.get(job.kind)
    if handler is None:
        raise RuntimeError(f"Tâche inconnue : {job.kind}")
    handler(Project.load(root / job.project), job, queue)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
    )
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    root = workspace()
    queue = default_queue()
    # Un arrêt brutal laisse des tâches marquées « en cours » qui ne le sont plus.
    if stale := queue.cancel_stale():
        logger.info("%d tâche(s) interrompue(s) remise(s) à plat", stale)
    logger.info("ouvrier prêt, projets dans %s", root)

    while not _stop:
        job = queue.claim()
        if job is None:
            time.sleep(IDLE_SLEEP)
            continue
        logger.info("tâche %d : %s sur %s", job.id, job.kind, job.project)
        try:
            execute(job, queue, root)
        except Exception as error:
            logger.exception("tâche %d en échec", job.id)
            queue.finish(job.id, error)
        else:
            queue.finish(job.id)
            logger.info("tâche %d terminée en %.0f s", job.id, queue.get(job.id).running_for)


if __name__ == "__main__":
    main()
