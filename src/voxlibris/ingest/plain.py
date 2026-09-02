"""Ingestion de texte brut ou Markdown.

C'est la porte d'entrée universelle : quand aucun lecteur spécialisé ne convient, il
reste toujours possible de coller le texte dans un fichier. C'est aussi la sortie de
secours d'un OCR fait ailleurs.

Le chapitrage suit les titres Markdown quand ils existent. À défaut, on cherche des
lignes isolées qui ressemblent à des titres de chapitre — courtes, sans ponctuation
finale, entourées de blancs. Sans rien de tel, le texte forme un chapitre unique, à
redécouper dans l'interface.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..document import Chapter, Document, clean_title

MARKDOWN_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*$")
# Un titre de chapitre en texte brut : bref, sans point final, souvent en capitales ou
# introduit par un mot-clé.
CHAPTER_WORD = re.compile(r"^\s*(chapitre|chapter|partie|livre|acte)\b", re.I)
MAX_HEADING_CHARS = 70

# Un texte tiré d'un PDF garde les pieds de page du livre imprimé. Personne ne les voit
# en relisant à l'écran, et le moteur les lit pourtant à voix haute, au milieu d'une
# phrase : « …des dédommagements. vingt-six CM deux P Page quarante-cinq. Malgré tout… »
#
# Deux formes se rencontrent. La ligne isolée ne portant qu'un numéro se reconnaît sans
# risque. La seconde est soudée au texte : un nombre, les restes du gabarit d'impression,
# puis « Page » et son numéro. On exige qu'elle suive une fin de phrase, commence par un
# chiffre et s'achève sur « Page N » — sans quoi une mention légitime, « voir à la
# page 45 », disparaîtrait elle aussi.
PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.I)
GLUED_FOOTER = re.compile(r"(?<=[.!?])\s*\d[0-9A-Za-z]{0,10}Page\s*\d{1,4}\b")


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    if stripped[-1] in ".,;:!?«»":
        return False
    return bool(CHAPTER_WORD.match(stripped)) or stripped.upper() == stripped


def _split_blocks(text: str) -> list[str]:
    blocks = [re.sub(r"\s*\n\s*", " ", b).strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return [cleaned for block in blocks if (cleaned := GLUED_FOOTER.sub("", block).strip())]


def ingest(path: Path, title: str = "", author: str = "Inconnu") -> Document:
    """Extrait un document depuis un fichier texte ou Markdown."""
    raw = path.read_text(encoding="utf-8", errors="replace")

    chapters: list[Chapter] = []
    current_title = ""
    buffer: list[str] = []
    source = "aucun repère"

    def flush() -> None:
        if not buffer:
            return
        # Les lignes sont recollées telles quelles : c'est `_split_blocks` qui décide
        # des paragraphes, à partir des lignes vides d'origine.
        body = "\n".join(buffer)
        chapters.append(
            Chapter(
                number=len(chapters) + 1,
                title=clean_title(current_title) or f"Chapitre {len(chapters) + 1}",
                paragraphs=_split_blocks(body),
            )
        )
        buffer.clear()

    lines = raw.splitlines()
    has_markdown = any(MARKDOWN_HEADING.match(line) for line in lines)
    dropped = sum(1 for line in lines if PAGE_NUMBER_LINE.match(line))
    lines = [line for line in lines if not PAGE_NUMBER_LINE.match(line)]

    for line in lines:
        heading = MARKDOWN_HEADING.match(line) if has_markdown else None
        if heading:
            source = "titres Markdown"
            flush()
            current_title = heading.group(2)
            continue
        if not has_markdown and _looks_like_heading(line) and buffer:
            source = "lignes de titre"
            flush()
            current_title = line.strip()
            continue
        buffer.append(line)

    flush()

    if not chapters:
        chapters = [Chapter(1, title or path.stem, _split_blocks(raw))]

    return Document(
        title=title or path.stem,
        author=author,
        chapters=chapters,
        source=path,
        needs_review=False,
        notes={"chapitrage": source, "numéros de page retirés": dropped},
    )
