"""Modèle de document commun à tous les formats d'entrée.

EPUB, PDF natif, scan et texte brut convergent vers cette structure, de sorte que tout
ce qui suit — relecture, normalisation, synthèse, assemblage — ignore d'où vient le
texte. Seul `needs_review` conserve la trace de son origine, parce qu'un texte issu d'un
OCR doit être relu et qu'un EPUB non.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


@dataclass
class Chapter:
    number: int
    title: str
    paragraphs: list[str] = field(default_factory=list)
    source_pages: tuple[int, int] | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_markdown(self) -> str:
        lines = ["---", f"chapter: {self.number}", f'title: "{self.title}"']
        if self.source_pages:
            lines.append(f"pages: {self.source_pages[0]}-{self.source_pages[1]}")
        lines += ["---", "", self.text, ""]
        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, text: str) -> Chapter:
        match = FRONT_MATTER.match(text)
        if not match:
            raise ValueError("Chapitre sans en-tête YAML")
        meta: dict[str, str] = {}
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
        body = text[match.end() :].strip()
        pages = None
        if "pages" in meta and "-" in meta["pages"]:
            first, _, last = meta["pages"].partition("-")
            pages = (int(first), int(last))
        return cls(
            number=int(meta["chapter"]),
            title=meta.get("title", ""),
            paragraphs=[p.strip() for p in body.split("\n\n") if p.strip()],
            source_pages=pages,
        )


@dataclass
class Document:
    title: str
    chapters: list[Chapter] = field(default_factory=list)
    author: str = "Inconnu"
    year: str | None = None
    language: str = "fr"
    source: Path | None = None
    # Vrai quand le texte vient d'un OCR : il comportera des coquilles qu'aucune
    # heuristique ne rattrape, et qu'il faut relire avant de les entendre.
    needs_review: bool = False
    # Ce que l'ingestion a mesuré ou deviné, pour affichage et diagnostic.
    notes: dict[str, object] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return sum(chapter.word_count for chapter in self.chapters)

    def write(self, directory: Path) -> list[Path]:
        """Écrit un fichier Markdown par chapitre et renvoie les chemins produits."""
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for chapter in self.chapters:
            path = directory / f"ch{chapter.number:02d}.md"
            path.write_text(chapter.to_markdown(), encoding="utf-8")
            written.append(path)
        return written

    @classmethod
    def read(cls, directory: Path, title: str = "", **kwargs) -> Document:
        chapters = [
            Chapter.from_markdown(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("ch*.md"))
        ]
        return cls(title=title or directory.name, chapters=chapters, **kwargs)


def clean_title(text: str) -> str:
    """Réduit un titre capté sur une page à une forme présentable.

    Les titres relevés sur un scan arrivent en capitales et souvent constellés
    d'ornements typographiques mal reconnus.
    """
    text = re.sub(r"\s+", " ", text).strip(" .:-—–_*#")
    if text and text.upper() == text:
        # Capitales intégrales : on repasse en casse de titre, en préservant les accents.
        text = text.capitalize()
    return text


def slugify(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^\w\s-]", "", ascii_text).strip()
