"""Assemble les WAV synthétisés en MP3 chapitrés et en un fichier M4B unique.

Le M4B est le format des lecteurs de livres audio : un seul fichier, des marqueurs de
chapitres, une couverture, et la reprise de lecture là où on s'était arrêté. Les MP3
individuels servent aux lecteurs qui ne le gèrent pas.

Les deux sorties passent d'abord par une normalisation de niveau EBU R128 à -19 LUFS,
valeur de référence pour la parole : sans elle, les écarts de volume d'un chapitre à
l'autre obligent l'auditeur à toucher au bouton toutes les dix minutes.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

LOUDNESS_LUFS = -19.0
LOUDNESS_PEAK = -3.0


@dataclass(frozen=True)
class BookMetadata:
    """Métadonnées gravées dans les tags des MP3 et du M4B."""

    title: str
    author: str = "Inconnu"
    year: str | None = None
    narrator: str | None = None

    @property
    def comment(self) -> str:
        parts = ["Synthèse vocale locale — voxlibris"]
        if self.narrator:
            parts.append(f"voix : {self.narrator}")
        return ", ".join(parts)


def run(*cmd: str | Path) -> None:
    subprocess.run([str(c) for c in cmd], check=True, capture_output=True)


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


def chapter_titles(segments_dir: Path) -> dict[int, str]:
    """Relit les titres depuis les segments, seule source qui suive la numérotation."""
    titles: dict[int, str] = {}
    for path in sorted(segments_dir.glob("ch*.jsonl")):
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        titles[int(first["chapter"])] = str(first["title"])
    return titles


def safe_name(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^\w\s-]", "", ascii_text).strip()


def extract_cover(source: Path | None, target: Path) -> Path | None:
    """Rend la couverture depuis la source, quel qu'en soit le format.

    L'import de PyMuPDF est local : l'assemblage doit rester utilisable sur un projet
    issu d'un EPUB ou d'un simple fichier texte, sans exiger la pile PDF.
    """
    if source is None or not source.exists():
        return None
    suffix = source.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png"}:
        shutil.copyfile(source, target)
        return target
    if suffix == ".pdf":
        import pymupdf

        pymupdf.open(source)[0].get_pixmap(dpi=200).save(target)
        return target
    if suffix == ".epub":
        from .ingest.metadata import cover

        if data := cover(source):
            target.write_bytes(data)
            return target
    return None


def normalized_wav(source: Path, target: Path) -> None:
    """Applique la normalisation de niveau EBU R128."""
    run(
        "ffmpeg",
        "-y",
        "-i",
        source,
        "-af",
        f"loudnorm=I={LOUDNESS_LUFS}:TP={LOUDNESS_PEAK}:LRA=7",
        "-ar",
        "24000",
        "-ac",
        "1",
        target,
    )


def write_mp3(
    wav: Path,
    target: Path,
    chapter: int,
    title: str,
    total: int,
    cover: Path | None,
    meta: BookMetadata,
) -> None:
    cmd: list[str | Path] = ["ffmpeg", "-y", "-i", wav]
    if cover:
        cmd += ["-i", cover, "-map", "0:a", "-map", "1:v", "-disposition:v", "attached_pic"]
    cmd += [
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "6",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-metadata",
        f"title={title}",
        "-metadata",
        f"artist={meta.author}",
        "-metadata",
        f"album={meta.title}",
        "-metadata",
        f"track={chapter}/{total}",
        "-metadata",
        "genre=Livre audio",
        "-metadata",
        f"comment={meta.comment}",
    ]
    if meta.year:
        cmd += ["-metadata", f"date={meta.year}"]
    cmd.append(target)
    run(*cmd)


def write_m4b(
    wavs: list[Path],
    titles: dict[int, str],
    target: Path,
    work: Path,
    cover: Path | None,
    meta: BookMetadata,
) -> None:
    """Concatène les chapitres et y attache les marqueurs de chapitres."""
    concat_list = work / "concat.txt"
    concat_list.write_text("".join(f"file '{w.resolve()}'\n" for w in wavs), encoding="utf-8")

    # Les timestamps doivent venir des durées réelles : une estimation décalerait
    # progressivement tous les marqueurs suivants.
    lines = [
        ";FFMETADATA1",
        f"title={meta.title}",
        f"artist={meta.author}",
        f"album={meta.title}",
        f"comment={meta.comment}",
    ]
    start_ms = 0
    for wav in wavs:
        chapter = int(wav.stem.removeprefix("ch"))
        end_ms = start_ms + int(probe_duration(wav) * 1000)
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={titles.get(chapter, wav.stem)}",
        ]
        start_ms = end_ms
    metadata = work / "chapters.txt"
    metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd: list[str | Path] = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list,
        "-i",
        metadata,
    ]
    maps = ["-map", "0:a"]
    if cover:
        cmd += ["-i", cover]
        maps += ["-map", "2:v", "-disposition:v", "attached_pic", "-c:v", "mjpeg"]
    cmd += [
        "-map_metadata",
        "1",
        *maps,
        "-codec:a",
        "aac",
        "-b:a",
        "64k",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        target,
    ]
    run(*cmd)


def build(
    wav_dir: Path,
    segments_dir: Path,
    out_dir: Path,
    meta: BookMetadata,
    cover_source: Path | None = None,
    write_mp3s: bool = True,
    on_progress: Callable[[str], None] = lambda _: None,
) -> Path:
    """Assemble les WAV en MP3 chapitrés et en un M4B, et renvoie le chemin du M4B."""
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} est introuvable dans le PATH.")

    sources = sorted(wav_dir.glob("ch*.wav"))
    if not sources:
        raise RuntimeError(f"Aucun WAV dans {wav_dir} — lancer la synthèse d'abord.")

    work = out_dir / "build"
    work.mkdir(parents=True, exist_ok=True)
    titles = chapter_titles(segments_dir)
    cover = extract_cover(cover_source, work / "cover.jpg")

    on_progress("Normalisation du niveau sonore (EBU R128)")
    normalized: list[Path] = []
    for source in sources:
        target = work / source.name
        normalized_wav(source, target)
        normalized.append(target)
        on_progress(f"  {source.name}  {probe_duration(target) / 60:5.1f} min")

    if write_mp3s:
        mp3_dir = out_dir / "mp3"
        mp3_dir.mkdir(parents=True, exist_ok=True)
        on_progress("Écriture des MP3 par chapitre")
        for wav in normalized:
            chapter = int(wav.stem.removeprefix("ch"))
            title = titles.get(chapter, wav.stem)
            name = f"{chapter:02d} - {safe_name(title)}.mp3"
            write_mp3(wav, mp3_dir / name, chapter, title, len(normalized), cover, meta)
            on_progress(f"  {name}")

    m4b = out_dir / f"{safe_name(meta.title) or 'audiobook'}.m4b"
    on_progress(f"Assemblage du M4B ({len(normalized)} chapitres)")
    write_m4b(normalized, titles, m4b, work, cover, meta)

    shutil.rmtree(work, ignore_errors=True)
    return m4b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", default="out/wav")
    parser.add_argument("--segments", default="work/segments")
    parser.add_argument("--out", default="out")
    parser.add_argument("--title", required=True)
    parser.add_argument("--author", default="Inconnu")
    parser.add_argument("--year")
    parser.add_argument("--narrator")
    parser.add_argument("--cover", help="PDF, JPEG ou PNG dont extraire la couverture")
    parser.add_argument("--skip-mp3", action="store_true")
    args = parser.parse_args()

    m4b = build(
        wav_dir=Path(args.wav),
        segments_dir=Path(args.segments),
        out_dir=Path(args.out),
        meta=BookMetadata(
            title=args.title, author=args.author, year=args.year, narrator=args.narrator
        ),
        cover_source=Path(args.cover) if args.cover else None,
        write_mp3s=not args.skip_mp3,
        on_progress=print,
    )
    total = probe_duration(m4b)
    print(
        f"\n{m4b}  —  {int(total // 3600)} h {int(total % 3600 // 60):02d} min, "
        f"{m4b.stat().st_size / 1e6:.1f} Mo"
    )


if __name__ == "__main__":
    main()
