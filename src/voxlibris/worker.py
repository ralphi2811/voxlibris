"""Atelier : exécute les tâches déposées par l'interface web.

Processus distinct de l'interface, pour deux raisons. La première est évidente : une
synthèse dure des dizaines de minutes et ne peut pas se dérouler dans une requête HTTP.
La seconde l'est moins et pèse plus lourd : seul cet atelier a besoin de CUDA et de la
pile PyTorch, ce qui laisse l'image de l'interface à quelques centaines de mégaoctets là
où celle-ci en pèse huit gigaoctets.

    python -m voxlibris.worker
"""

from __future__ import annotations

import json
import logging
import signal
import time
from pathlib import Path

from . import atelier
from .assemble import build
from .config import setting
from .normalize import build_segments
from .project import Project
from .web.jobs import Cancelled, Job, Queue, default_queue, workspace

logger = logging.getLogger("voxlibris.worker")
IDLE_SLEEP = 1.0

_stop = False


def _request_stop(*_) -> None:
    global _stop
    _stop = True
    logger.info("arrêt demandé : la tâche en cours va s'achever avant de rendre la main")


def run_normalize(project: Project, job: Job, queue: Queue) -> None:
    if (scale := job.params.get("pause_scale")) is not None:
        project.pause_scale = float(scale)
    if (announce := job.params.get("announce")) is not None:
        project.announce_chapters = bool(announce)
    project.save()
    counts = build_segments(
        project.text_dir,
        project.segments_dir,
        announce_chapters=project.announce_chapters,
        pause_scale=project.pause_scale,
    )
    queue.report(
        job.id,
        1.0,
        f"{sum(counts.values())} segments écrits, silences à {project.pause_scale:g}×",
    )


def track_is_current(track: Path, segments: Path) -> bool:
    """Vrai si la piste existe et date d'après ses segments.

    Cas vécu : un passage corrigé dans le texte, les segments repréparés, et la synthèse
    qui répond « déjà synthétisé » en gardant la piste d'avant — la correction n'a jamais
    atteint l'audio. La date des segments ne bouge que quand leur contenu change.
    """
    return track.exists() and track.stat().st_mtime >= segments.stat().st_mtime


def run_synth(project: Project, job: Job, queue: Queue) -> None:
    from .tts.backends import load
    from .tts.synth import load_segments, previous_takes, profile_for_voice, synthesize_chapter

    backend = job.params.get("backend", "xtts")
    voice = job.params.get("voice")
    force = bool(job.params.get("force"))
    # Le périphérique par défaut vient de l'environnement : un conteneur sans GPU le
    # fixe à « cpu », et Piper comme Kokoro y tournent sans rien changer d'autre.
    device = job.params.get("device") or setting("VOXLIBRIS_DEVICE", "cuda")

    paths = sorted(project.segments_dir.glob("ch*.jsonl"))
    if not paths:
        raise RuntimeError("Aucun segment : lancer la normalisation d'abord.")

    # Le réglage précédent est relevé avant d'appliquer le nouveau : c'est leur écart qui
    # décide s'il faut tout refaire. La vitesse compte au même titre que la voix — un
    # livre dont la moitié des chapitres accélère serait pire qu'un livre trop rapide.
    previous = f"{project.backend}/{project.voice}@{project.speed:.2f}"
    if (wanted := job.params.get("speed")) is not None:
        project.speed = float(wanted)

    queue.report(job.id, 0.0, f"chargement du moteur {backend} à {project.speed:g}×")
    engine = load(backend, voice, device, project.speed)
    chosen = getattr(engine, "voice", backend)

    signature = f"{backend}/{chosen}@{project.speed:.2f}"
    changed = project.voice is not None and previous != signature
    if changed:
        queue.report(job.id, message="réglage différent du précédent : tout est resynthétisé")
    project.backend, project.voice = backend, chosen
    project.save()

    if not engine.supports_speed and project.speed != 1.0:
        queue.report(
            job.id,
            message=f"le moteur {backend} ne sait pas moduler le débit : vitesse ignorée",
        )
    if warning := remote_cost_warning(backend, paths):
        queue.report(job.id, message=warning)
    for line in moderation_warnings(backend, paths):
        queue.report(job.id, message=line)

    profile = profile_for_voice(engine, load_segments(paths[0]), project.calibration_file)
    queue.report(job.id, message=f"débit calibré à {profile.chars_per_second} car/s")

    def check_cancel(*_) -> None:
        # Relevé entre deux segments : la piste en cours n'est écrite qu'une fois complète,
        # un arrêt ne laisse donc jamais de chapitre à moitié fait.
        if queue.cancel_requested(job.id):
            raise Cancelled()

    warnings: list[str] = []
    for index, path in enumerate(paths):
        check_cancel()
        target = project.wav_dir / f"{path.stem}.wav"
        if track_is_current(target, path) and not (force or changed):
            queue.report(job.id, (index + 1) / len(paths), f"{path.stem} déjà synthétisé")
            continue
        reuse = None
        if target.exists() and not (force or changed):
            # Même voix, même débit : ce qui n'a pas changé de texte est repris tel quel.
            number = int(path.stem.removeprefix("ch"))
            reuse = previous_takes(target, project.timing(number))
            queue.report(job.id, message=f"{path.stem} : segments plus récents que la piste")
        segments = load_segments(path)
        queue.report(
            job.id,
            index / len(paths),
            f"{path.stem} — {segments[0].title} ({len(segments)} segments)",
        )
        result = synthesize_chapter(engine, segments, profile, on_segment=check_cancel, reuse=reuse)
        result.write(target)
        warnings += result.warnings
        summary = f"{path.stem} → {result.duration / 60:.1f} min"
        if result.reused:
            summary += f", {result.reused} segments repris de la piste précédente"
        queue.report(job.id, (index + 1) / len(paths), summary)

    if warnings:
        queue.report(job.id, message=f"{len(warnings)} segment(s) à vérifier :")
        for warning in warnings:
            queue.report(job.id, message=f"  {warning}")
    else:
        queue.report(job.id, message="contrôle qualité : aucun segment hors tolérance")


def remote_cost_warning(backend: str, paths: list[Path]) -> str:
    """Annonce la dépense avant de l'engager, pour un moteur facturé à l'usage.

    Une synthèse dure des dizaines de minutes : découvrir la facture après coup serait
    une mauvaise surprise, et le calcul ne coûte qu'une lecture des segments.
    """
    if backend != "voxtral":
        return ""
    from .tts.voxtral import Client, estimate_cost

    # Servi sur la machine, le même moteur ne facture rien : l'annonce serait fausse.
    if Client().is_local:
        return ""

    characters = sum(
        len(json.loads(line)["text"]) for path in paths for line in path.open(encoding="utf-8")
    )
    return (
        f"moteur distant : {characters:,} caractères seront envoyés à Mistral, "
        f"soit environ {estimate_cost(characters):.2f} $".replace(",", " ")
    )


def moderation_warnings(backend: str, paths: list[Path], show: int = 5) -> list[str]:
    """Signale d'avance les phrases que la modération du service va refuser.

    Le contrôle porte sur le livre entier, coûte quelques secondes et une fraction de
    centime — à comparer à une synthèse de vingt minutes interrompue à mi-parcours, les
    segments déjà produits ayant été facturés.

    Il ne bloque rien : c'est un avertissement. Sur une biographie de Louis Braille, le
    classificateur a refusé le passage décrivant les préjugés d'un personnage envers les
    aveugles, et pris « Je m'appelle Gabriel Gautier » pour une donnée personnelle. On ne
    va pas empêcher quelqu'un de faire lire son livre pour cela.
    """
    if backend != "voxtral":
        return []
    from .tts.voxtral import Client, VoxtralError

    # Le serveur local n'a pas de filtre de modération — c'est même tout son intérêt.
    if Client().is_local:
        return []

    segments = [json.loads(line) for path in paths for line in path.open(encoding="utf-8")]
    try:
        verdicts = Client().moderate([s["text"] for s in segments])
    except VoxtralError as error:
        return [f"contrôle de modération impossible : {error}"]

    refused = [
        (segment, categories)
        for segment, categories in zip(segments, verdicts, strict=False)
        if categories
    ]
    if not refused:
        return ["modération : aucun segment signalé"]

    lines = [
        f"[modération] {len(refused)} segment(s) sur {len(segments)} risquent d'être "
        "refusés par Mistral ; ils seront remplacés par un silence et listés en fin de tâche"
    ]
    for segment, categories in refused[:show]:
        lines.append(
            f"  ch{segment['chapter']:02d} #{segment['idx']} [{', '.join(categories)}] "
            f"« {segment['text'][:70]} »"
        )
    if len(refused) > show:
        lines.append(f"  … et {len(refused) - show} autre(s)")
    return lines


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
        if queue.cancel_requested(job.id):
            raise Cancelled()
        backend, voice = choice["backend"], choice.get("voice")
        queue.report(job.id, index / len(choices), f"{backend} / {voice}")
        try:
            device = job.params.get("device") or setting("VOXLIBRIS_DEVICE", "cuda")
            engine = load(backend, voice, device)
            result = synthesize_chapter(engine, segments, QualityProfile())
            name = f"{backend}--{(voice or 'defaut').replace(' ', '_')}.wav"
            result.write(target_dir / name)
            queue.report(job.id, message=f"  → {name}")
        except Exception as error:  # une voix indisponible n'arrête pas la comparaison
            queue.report(job.id, message=f"  échec {backend}/{voice} : {error}")
    queue.report(job.id, 1.0, "échantillons prêts")


def run_resynth(project: Project, job: Job, queue: Queue) -> None:
    """Rejoue un seul segment et le recolle dans sa piste, sans refaire le chapitre.

    C'est le geste que le manifeste de minutage rend possible : on sait où commence et où
    finit chaque segment dans le fichier. On remplace cette plage, on décale ce qui suit,
    et le M4B n'a plus qu'à être réassemblé — quelques secondes, contre vingt minutes.
    """
    import numpy as np
    import soundfile as sf

    from .tts.backends import load
    from .tts.quality import SAMPLE_RATE
    from .tts.synth import load_segments, profile_for_voice, synthesize_chapter

    number, idx = int(job.params["chapter"]), int(job.params["idx"])
    segments_path = project.segments_dir / f"ch{number:02d}.jsonl"
    track = project.wav_dir / f"ch{number:02d}.wav"
    timing = project.timing(number)
    if not (segments_path.exists() and track.exists() and timing):
        raise RuntimeError("Piste ou segments absents : synthétisez d'abord le chapitre.")
    wanted = [s for s in load_segments(segments_path) if s.idx == idx]
    position = next((i for i, e in enumerate(timing) if e["idx"] == idx), None)
    if not wanted or position is None:
        raise RuntimeError(f"Segment {idx} introuvable dans le chapitre {number}.")
    if not project.backend:
        raise RuntimeError("Aucune voix retenue pour ce projet.")

    device = job.params.get("device") or setting("VOXLIBRIS_DEVICE", "cuda")
    queue.report(job.id, 0.1, f"chargement du moteur {project.backend} à {project.speed:g}×")
    engine = load(project.backend, project.voice, device, project.speed)
    profile = profile_for_voice(engine, load_segments(segments_path), project.calibration_file)
    queue.report(job.id, 0.5, f"ch{number:02d} segment {idx} — « {wanted[0].text[:60]} »")
    result = synthesize_chapter(engine, wanted, profile)

    audio, _ = sf.read(track, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    entry = timing[position]
    start = int(round(float(entry["start"]) * SAMPLE_RATE))
    following = timing[position + 1]["start"] if position + 1 < len(timing) else None
    stop = int(round(float(following) * SAMPLE_RATE)) if following is not None else len(audio)
    patched = np.concatenate([audio[:start], result.audio, audio[stop:]])
    delta = (len(result.audio) - (stop - start)) / SAMPLE_RATE

    fresh = result.timing[0]
    entry.update(
        end=round(float(entry["start"]) + (fresh["end"] - fresh["start"]), 2),
        attempts=fresh["attempts"],
        clean=fresh["clean"],
        cause=fresh.get("cause", ""),
        split=fresh["split"],
        approved=False,
    )
    for later in timing[position + 1 :]:
        later["start"] = round(float(later["start"]) + delta, 2)
        later["end"] = round(float(later["end"]) + delta, 2)

    sf.write(track, patched, SAMPLE_RATE)
    track.with_suffix(".timing.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    verdict = "propre" if fresh["clean"] else f"toujours signalé ({fresh.get('cause')})"
    queue.report(
        job.id, 1.0, f"segment {idx} rejoué : {verdict}. Réassemblez pour mettre le M4B à jour."
    )


def run_assemble(project: Project, job: Job, queue: Queue) -> None:
    cover = project.cover_source
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
    "resynth": run_resynth,
    "assemble": run_assemble,
}


def execute(job: Job, queue: Queue, root: Path) -> None:
    handler = HANDLERS.get(job.kind)
    if handler is None:
        raise RuntimeError(f"Tâche inconnue : {job.kind}")
    handler(Project.load(root / job.project), job, queue)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    root = workspace()
    queue = default_queue()
    # Un seul atelier tourne à la fois : au démarrage, toute tâche encore marquée
    # « en cours » a été interrompue par l'arrêt précédent, quel que soit son âge.
    if stale := queue.cancel_stale(older_than=0):
        logger.info("%d tâche(s) interrompue(s) remise(s) à plat", stale)
    pulse = atelier.Pulse(root)
    pulse.start()
    logger.info("atelier prêt, projets dans %s", root)

    while not _stop:
        job = queue.claim()
        if job is None:
            time.sleep(IDLE_SLEEP)
            continue
        logger.info("tâche %d : %s sur %s", job.id, job.kind, job.project)
        pulse.busy = job.id
        try:
            execute(job, queue, root)
        except Cancelled as stop:
            logger.info("tâche %d arrêtée à la demande", job.id)
            queue.finish(job.id, stop)
        except Exception as error:
            logger.exception("tâche %d en échec", job.id)
            queue.finish(job.id, error)
        else:
            queue.finish(job.id)
            logger.info("tâche %d terminée en %.0f s", job.id, queue.get(job.id).running_for)
        finally:
            pulse.busy = None
    pulse.stop()


if __name__ == "__main__":
    main()
