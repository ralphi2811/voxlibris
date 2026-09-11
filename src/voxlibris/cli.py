"""Ligne de commande de voxlibris.

Chaque étape du pipeline est une commande, et chacune reprend là où la précédente s'est
arrêtée. L'interface web n'est qu'une façade sur ces mêmes fonctions : tout ce qu'elle
permet est faisable ici, et l'inverse aussi.

    voxlibris ingest livre.epub projets/mon-livre
    voxlibris review projets/mon-livre
    voxlibris normalize projets/mon-livre
    voxlibris synth projets/mon-livre --backend xtts --voice "Viktor Menelaos"
    voxlibris assemble projets/mon-livre

`run` enchaîne le tout, ce qui n'a de sens que pour une source sans relecture — un EPUB,
un PDF natif — et la commande refuse de le faire pour un scan.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import ingest as ingest_module
from .project import Project

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


def _open(root: Path) -> Project:
    try:
        return Project.load(root)
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error)) from error


@app.command()
def ingest(
    source: Path = typer.Argument(..., help="EPUB, PDF ou fichier texte"),
    root: Path = typer.Argument(..., help="dossier du projet à créer"),
    title: str = typer.Option("", help="remplace le titre lu dans le fichier"),
    author: str = typer.Option("", help="remplace l'auteur lu dans le fichier"),
) -> None:
    """Extrait le texte d'un livre et crée le projet."""
    kwargs = {k: v for k, v in {"title": title, "author": author}.items() if v}
    document = ingest_module.ingest(source, **kwargs)
    project = Project.create(root, document)

    table = Table("Chapitre", "Mots", "Titre", title=f"{document.title} — {document.author}")
    for chapter in document.chapters:
        words = f"{chapter.word_count:,}".replace(",", " ")
        table.add_row(str(chapter.number), words, chapter.title)
    console.print(table)
    console.print(f"Format détecté : [bold]{document.notes.get('kind')}[/bold]")
    console.print(f"{document.word_count:,}".replace(",", " ") + " mots au total.")

    if document.needs_review:
        console.print(
            "\n[yellow]Ce texte vient d'une reconnaissance de caractères et doit être "
            f"relu.[/yellow]\nCorrigez [bold]{project.raw_dir}[/bold] vers "
            f"[bold]{project.clean_dir}[/bold], en vous aidant de « voxlibris review »."
        )
    else:
        console.print("\n[green]Aucune relecture nécessaire.[/green]")


@app.command()
def review(
    root: Path,
    vocabulary: Path = typer.Option(None, help="fichier de mots propres à l'ouvrage"),
) -> None:
    """Signale les formes suspectes d'un texte océrisé."""
    from .review import inspect_directory

    project = _open(root)
    words = vocabulary.read_text(encoding="utf-8").split() if vocabulary else []
    suspects = inspect_directory(project.text_dir, language=project.language, vocabulary=words)

    current = ""
    for suspect in suspects:
        if suspect.chapter != current:
            current = suspect.chapter
            console.print(f"\n[bold]{current}[/bold]")
        console.print(f"  [cyan]{suspect.word:<18}[/cyan] [{suspect.reason}]  …{suspect.context}…")
    console.print(f"\n{len(suspects)} formes à vérifier.")


@app.command("drop-chapter")
def drop_chapter(
    root: Path,
    number: int = typer.Argument(..., help="numéro du chapitre à retirer"),
) -> None:
    """Retire un chapitre — page de copyright, annexe — et renumérote les suivants."""
    project = _open(root)
    try:
        stale = project.delete_chapter(number)
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error)) from error

    console.print(f"Chapitre {number} retiré ; les suivants ont reculé d'un rang.")
    console.print(
        f"Écarté car devenu faux : {stale['segments']} fichier(s) de segments, "
        f"{stale['pistes']} piste(s), {stale['assemblages']} assemblage(s).\n"
        "[yellow]Repassez par « normalize » puis « synth ».[/yellow]"
    )

    table = Table("Chapitre", "Mots", "Titre")
    for state in project.chapter_states():
        table.add_row(str(state["number"]), str(state["words"]), str(state["title"]))
    console.print(table)


@app.command("merge-chapter")
def merge_chapter(
    root: Path,
    number: int = typer.Argument(..., help="chapitre à recoller au précédent"),
) -> None:
    """Recolle un fragment au chapitre précédent — illustration ayant coupé le texte."""
    project = _open(root)
    try:
        project.merge_into_previous(number)
    except (FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error

    console.print(f"Chapitre {number} recollé au {number - 1}.")
    table = Table("Chapitre", "Mots", "Titre")
    for state in project.chapter_states():
        table.add_row(str(state["number"]), str(state["words"]), str(state["title"]) or "—")
    console.print(table)


@app.command("llm-check")
def llm_check() -> None:
    """Vérifie que le modèle de langage configuré répond."""
    from .llm import LLM, LLMConfig, LLMError

    config = LLMConfig.from_env()
    console.print(f"Modèle  : [bold]{config.model}[/bold]")
    console.print(f"Adresse : {config.base_url}")
    if not config.enabled:
        console.print("[yellow]Assistance désactivée (VOXLIBRIS_LLM_ENABLED=0).[/yellow]")
        return
    try:
        console.print(f"[green]{LLM(config).probe()}[/green]")
    except LLMError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error


@app.command()
def proofread(
    root: Path,
    vocabulary: Path = typer.Option(None, help="fichier de mots propres à l'ouvrage"),
) -> None:
    """Propose des corrections pour les formes suspectes, sans rien appliquer."""
    from .proofread import suggest

    project = _open(root)
    words = vocabulary.read_text(encoding="utf-8").split() if vocabulary else []
    report = suggest(project.chapter_texts(), language=project.language, vocabulary=words)
    if report.error:
        console.print(f"[red]{report.error}[/red]")

    table = Table("Chapitre", "Forme", "Proposition", "Motif du signalement")
    for suggestion in report.suggestions:
        table.add_row(
            suggestion.chapter,
            suggestion.word,
            f"[green]{suggestion.replacement}[/green]",
            suggestion.reason,
        )
    console.print(table)

    project.suggestions_file.parent.mkdir(parents=True, exist_ok=True)
    project.suggestions_file.write_text(report.to_json(), encoding="utf-8")
    console.print(
        f"{len(report.suggestions)} proposition(s) sur {report.asked} forme(s) soumises, "
        f"{report.suspects} signalées au total. {len(report.rejected)} écartée(s) par les "
        f"garde-fous.\nRien n'a été modifié : voir {project.suggestions_file}."
    )


@app.command("bench-proofread")
def bench_proofread(
    raw: Path = typer.Argument(..., help="dossier du texte océrisé brut"),
    clean: Path = typer.Argument(..., help="le même texte, relu à la main"),
    recurrence_limit: int = typer.Option(4, help="seuil au-delà duquel une forme est laissée"),
    show: int = typer.Option(12, help="nombre d'exemples affichés par catégorie"),
) -> None:
    """Mesure les suggestions contre une relecture humaine de référence."""
    from .bench import evaluate, read_bodies

    outcome = evaluate(read_bodies(raw), read_bodies(clean), recurrence_limit=recurrence_limit)
    if outcome.report.error:
        console.print(f"[red]{outcome.report.error}[/red]")

    table = Table("Mesure", "Nombre", title="Suggestions contre relecture humaine")
    table.add_row("Corrections faites à la main", str(outcome.corrections))
    table.add_row("(apostrophes redressées, hors sujet)", str(outcome.typographic))
    table.add_row("Formes signalées", str(outcome.report.suspects))
    table.add_row("Formes soumises au modèle", str(outcome.report.asked))
    table.add_row("[green]Fautes corrigées[/green]", str(len(outcome.fixed)))
    table.add_row("Corrections inexactes", str(len(outcome.inexact)))
    table.add_row("[red]Modifications indues[/red]", str(len(outcome.undue)))
    table.add_row("Fautes manquées (signalées)", str(len(outcome.missed_flagged)))
    table.add_row("Fautes manquées (non signalées)", str(len(outcome.missed_unflagged)))
    table.add_row("Propositions écartées d'office", str(len(outcome.report.rejected)))
    console.print(table)
    console.print(
        f"Précision {outcome.precision:.0%} — rappel {outcome.recall:.0%}\n"
        "La ligne qui décide est « modifications indues » : ce sont des propositions "
        "portant sur un texte déjà juste."
    )

    groups = (
        ("Corrigé", [(c.word, s.replacement, c.expected) for c, s in outcome.fixed]),
        ("Inexact", [(c.word, s.replacement, c.expected) for c, s in outcome.inexact]),
        ("[red]Indu[/red]", [(s.word, s.replacement, "rien à corriger") for s in outcome.undue]),
    )
    for title, rows in groups:
        if not rows:
            continue
        console.print(f"\n[bold]{title}[/bold]")
        for word, proposal, expected in rows[:show]:
            console.print(f"  {word:<20} → {proposal:<20} (attendu : {expected})")
        if len(rows) > show:
            console.print(f"  … et {len(rows) - show} autre(s)")


@app.command()
def normalize(
    root: Path,
    pauses: float = typer.Option(
        None, help="étire tous les silences (1.3 pour une ponctuation plus marquée)"
    ),
) -> None:
    """Prépare les segments à synthétiser à partir du texte faisant foi."""
    from .normalize import build_segments

    project = _open(root)
    if pauses is not None:
        project.pause_scale = pauses
        project.save()
    counts = build_segments(project.text_dir, project.segments_dir, pause_scale=project.pause_scale)
    total = sum(counts.values())
    console.print(
        f"{total} segments écrits dans {project.segments_dir} (silences à {project.pause_scale:g}×)"
    )


@app.command()
def voices(
    backend: str = typer.Argument("xtts", help="xtts, kokoro, piper, voxtral ou zonos2"),
) -> None:
    """Liste les voix disponibles pour un moteur."""
    from .tts.backends import load

    try:
        found = load(backend).voices()
    except Exception as error:
        # Moteur non installé, clé absente, service injoignable : autant de causes
        # ordinaires, dont le message importe bien plus que la pile d'appels.
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    for voice in found:
        console.print(voice)


@app.command()
def synth(
    root: Path,
    backend: str = typer.Option("xtts", help="xtts, kokoro ou piper"),
    voice: str = typer.Option(None, help="voix du moteur ; défaut selon le moteur"),
    chapters: str = typer.Option(None, help="liste de numéros, par exemple 1,2,3"),
    force: bool = typer.Option(False, help="resynthétiser même si le WAV existe"),
    speed: float = typer.Option(None, help="débit de parole ; 0.9 pour ralentir"),
    device: str = typer.Option("cuda", help="cuda ou cpu"),
) -> None:
    """Synthétise les chapitres, avec contrôle qualité et reprise des ratés."""
    from .tts.backends import load
    from .tts.synth import load_segments, profile_for_voice, synthesize_chapter

    project = _open(root)
    paths = sorted(project.segments_dir.glob("ch*.jsonl"))
    if not paths:
        raise typer.BadParameter("Aucun segment : lancez d'abord « voxlibris normalize ».")
    if chapters:
        wanted = {int(c) for c in chapters.split(",")}
        paths = [p for p in paths if int(p.stem.removeprefix("ch")) in wanted]

    previous = f"{project.backend}/{project.voice}@{project.speed:.2f}"
    if speed is not None:
        project.speed = speed

    engine = load(backend, voice, device, project.speed)
    chosen = getattr(engine, "voice", backend)

    # Un changement de voix ou de vitesse invalide tout : sans cela le livre changerait
    # de narrateur — ou de débit — en cours de route, sans le moindre avertissement.
    signature = f"{backend}/{chosen}@{project.speed:.2f}"
    changed = project.voice is not None and previous != signature
    if changed:
        console.print("[yellow]Réglage différent du précédent : tout est resynthétisé.[/yellow]")
    project.backend, project.voice = backend, chosen
    project.save()

    profile = profile_for_voice(engine, load_segments(paths[0]), project.calibration_file)
    console.print(
        f"Moteur [bold]{backend}[/bold], voix [bold]{chosen}[/bold], "
        f"vitesse {project.speed:g}×, débit calibré à {profile.chars_per_second} car/s.\n"
    )

    warnings: list[str] = []
    for path in paths:
        target = project.wav_dir / f"{path.stem}.wav"
        if target.exists() and not (force or changed):
            console.print(f"{path.stem} : déjà synthétisé.")
            continue
        segments = load_segments(path)
        console.print(f"{path.stem} — {segments[0].title} ({len(segments)} segments)")
        result = synthesize_chapter(engine, segments, profile)
        result.write(target)
        console.print(f"  → {target.name}  {result.duration / 60:.1f} min")
        warnings += result.warnings

    if warnings:
        console.print(f"\n[yellow]{len(warnings)} segment(s) à vérifier :[/yellow]")
        for warning in warnings:
            console.print(f"  {warning}")
    else:
        console.print("\n[green]Contrôle qualité : aucun segment hors tolérance.[/green]")


@app.command()
def assemble(
    root: Path,
    skip_mp3: bool = typer.Option(False, help="ne produire que le M4B"),
) -> None:
    """Assemble les chapitres en MP3 et en un M4B chapitré."""
    from .assemble import build, probe_duration

    project = _open(root)
    cover = Path(project.source) if project.source else None
    m4b = build(
        wav_dir=project.wav_dir,
        segments_dir=project.segments_dir,
        out_dir=project.out_dir,
        meta=project.metadata,
        cover_source=cover,
        write_mp3s=not skip_mp3,
        on_progress=console.print,
    )
    total = probe_duration(m4b)
    console.print(
        f"\n[green]{m4b}[/green] — {int(total // 3600)} h {int(total % 3600 // 60):02d} min, "
        f"{m4b.stat().st_size / 1e6:.1f} Mo"
    )


@app.command()
def status(root: Path) -> None:
    """Montre où en est un projet."""
    project = _open(root)
    console.print(f"[bold]{project.title}[/bold] — {project.author}")
    for key, value in project.status().items():
        console.print(f"  {key:<24} {value}")


@app.command()
def run(
    source: Path,
    root: Path,
    backend: str = typer.Option("xtts"),
    voice: str = typer.Option(None),
    device: str = typer.Option("cuda"),
) -> None:
    """Enchaîne toute la chaîne, pour une source ne demandant pas de relecture."""
    document = ingest_module.ingest(source)
    if document.needs_review:
        raise typer.BadParameter(
            "Ce texte vient d'un OCR et doit être relu : « run » n'a pas de sens ici. "
            "Passez par ingest, review, puis normalize."
        )
    Project.create(root, document)
    normalize(root, pauses=None)
    synth(
        root,
        backend=backend,
        voice=voice,
        chapters=None,
        force=False,
        speed=None,
        device=device,
    )
    assemble(root, skip_mp3=False)


if __name__ == "__main__":
    app()
