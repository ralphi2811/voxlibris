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
    suspects = inspect_directory(
        project.text_dir, language=project.language, vocabulary=words
    )

    current = ""
    for suspect in suspects:
        if suspect.chapter != current:
            current = suspect.chapter
            console.print(f"\n[bold]{current}[/bold]")
        console.print(f"  [cyan]{suspect.word:<18}[/cyan] [{suspect.reason}]  …{suspect.context}…")
    console.print(f"\n{len(suspects)} formes à vérifier.")


@app.command()
def normalize(root: Path) -> None:
    """Prépare les segments à synthétiser à partir du texte faisant foi."""
    from .normalize import build_segments

    project = _open(root)
    counts = build_segments(project.text_dir, project.segments_dir)
    total = sum(counts.values())
    console.print(f"{total} segments écrits dans {project.segments_dir}")


@app.command()
def voices(backend: str = typer.Argument("xtts", help="xtts, kokoro ou piper")) -> None:
    """Liste les voix disponibles pour un moteur."""
    from .tts.backends import load

    for voice in load(backend).voices():
        console.print(voice)


@app.command()
def synth(
    root: Path,
    backend: str = typer.Option("xtts", help="xtts, kokoro ou piper"),
    voice: str = typer.Option(None, help="voix du moteur ; défaut selon le moteur"),
    chapters: str = typer.Option(None, help="liste de numéros, par exemple 1,2,3"),
    force: bool = typer.Option(False, help="resynthétiser même si le WAV existe"),
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

    engine = load(backend, voice, device)
    chosen = getattr(engine, "voice", backend)

    # Un changement de voix invalide tout : sans cela le livre changerait de narrateur
    # en cours de route, sans le moindre avertissement.
    signature = f"{backend}/{chosen}"
    changed = project.voice is not None and f"{project.backend}/{project.voice}" != signature
    if changed:
        console.print("[yellow]Voix différente de la précédente : tout est resynthétisé.[/yellow]")
    project.backend, project.voice = backend, chosen
    project.save()

    profile = profile_for_voice(
        engine, load_segments(paths[0]), project.calibration_file
    )
    console.print(
        f"Moteur [bold]{backend}[/bold], voix [bold]{chosen}[/bold], "
        f"débit calibré à {profile.chars_per_second} car/s.\n"
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
    normalize(root)
    synth(root, backend=backend, voice=voice, chapters=None, force=False, device=device)
    assemble(root, skip_mp3=False)


if __name__ == "__main__":
    app()
