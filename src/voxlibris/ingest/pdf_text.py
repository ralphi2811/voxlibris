"""Extraction du texte d'un PDF nativement textuel.

À distinguer d'un scan : ici le texte a été produit par un traitement de texte, il est
exact, et il n'y a rien à relire. Restent deux difficultés, qui n'ont rien à voir avec
celles d'un scan :

- **Retrouver les chapitres.** Un PDF ne connaît que des pages. Les signets, quand ils
  existent, donnent la réponse exacte ; sinon il faut la déduire des tailles de police,
  un titre étant composé plus grand que le corps.
- **Recomposer les paragraphes.** Un PDF ne stocke que des lignes positionnées. Les
  regrouper demande de repérer les alinéas et les fins de paragraphe.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from ..document import Chapter, Document, clean_title
from .layout import Layout, detect

# Au-delà de ce rapport à la taille du corps, une ligne est tenue pour un titre.
HEADING_RATIO = 1.25
MIN_CHAPTER_CHARS = 200


def _page_lines(page: pymupdf.Page, layout: Layout) -> list[tuple[str, float, float]]:
    """Renvoie (texte, taille, x) pour chaque ligne du corps de la page."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            y = line["bbox"][1]
            if not (layout.header_y <= y < layout.footer_y):
                continue
            text = "".join(s["text"] for s in spans).strip()
            size = max(s["size"] for s in spans)
            if text:
                lines.append((text, size, line["bbox"][0]))
    return lines


def _paragraphs(lines: list[tuple[str, float, float]], layout: Layout) -> list[str]:
    """Recompose les paragraphes à partir des lignes positionnées."""
    paragraphs: list[list[str]] = []
    for text, _, x in lines:
        indented = x > layout.margin_left + 4
        starts_dialogue = text.startswith(("—", "–", "«"))
        if not paragraphs or indented or starts_dialogue:
            paragraphs.append([text])
        else:
            paragraphs[-1].append(text)

    joined = []
    for parts in paragraphs:
        merged = ""
        for part in parts:
            if merged.endswith("-"):
                merged = merged[:-1] + part.lstrip()
            elif merged:
                merged += " " + part
            else:
                merged = part
        joined.append(merged)
    return joined


def _outline_starts(doc: pymupdf.Document) -> list[tuple[int, str]]:
    """Chapitres déclarés par les signets du PDF : (page 0-indexée, titre)."""
    try:
        toc = doc.get_toc()
    except Exception:
        return []
    # Seul le premier niveau nous intéresse : les sous-sections ne sont pas des pistes.
    tops = [(page - 1, title) for level, title, page in toc if level == 1 and page > 0]
    return sorted(set(tops))


def _heading_starts(doc: pymupdf.Document, layout: Layout) -> list[tuple[int, str]]:
    """Repli sans signets : les lignes nettement plus grandes que le corps."""
    threshold = layout.body_size * HEADING_RATIO
    starts: list[tuple[int, str]] = []
    for number in range(len(doc)):
        for text, size, _ in _page_lines(doc[number], layout):
            if size >= threshold and len(text) < 90:
                starts.append((number, text))
                break
    return starts


def ingest(path: Path, title: str = "", author: str = "") -> Document:
    """Extrait un document depuis un PDF nativement textuel."""
    doc = pymupdf.open(path)
    layout = detect(doc)

    starts = _outline_starts(doc)
    source = "signets"
    if not starts:
        starts = _heading_starts(doc, layout)
        source = "tailles de police"
    if not starts:
        starts = [(0, title or path.stem)]
        source = "aucun repère"

    chapters: list[Chapter] = []
    for index, (first_page, heading) in enumerate(starts):
        last_page = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(doc) - 1
        lines: list[tuple[str, float, float]] = []
        for number in range(first_page, last_page + 1):
            lines += _page_lines(doc[number], layout)

        # La ligne de titre est déjà l'intitulé du chapitre : elle n'a pas à être lue
        # une seconde fois au début du texte.
        if lines and lines[0][0].strip() == heading.strip():
            lines = lines[1:]

        paragraphs = _paragraphs(lines, layout)
        if sum(len(p) for p in paragraphs) < MIN_CHAPTER_CHARS:
            continue
        chapters.append(
            Chapter(
                number=len(chapters) + 1,
                title=clean_title(heading) or f"Chapitre {len(chapters) + 1}",
                paragraphs=paragraphs,
                source_pages=(first_page + 1, last_page + 1),
            )
        )

    metadata = doc.metadata or {}
    return Document(
        title=title or metadata.get("title") or path.stem,
        author=author or metadata.get("author") or "Inconnu",
        chapters=chapters,
        source=path,
        needs_review=False,
        notes={"layout": layout.describe(), "chapitrage": source},
    )
