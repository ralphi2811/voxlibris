"""Extraction du texte d'un EPUB.

C'est le format le plus favorable, et de loin : la structure en chapitres y est déjà
déclarée, le texte est du texte — pas une image reconnue au jugé — et il n'y a donc
**aucune relecture** à prévoir. Un EPUB traverse toute la chaîne sans intervention.

Deux sources de structure coexistent dans un EPUB, et elles ne se recouvrent pas :

- le **spine** donne l'ordre de lecture, c'est-à-dire la suite exacte des fichiers ;
- la **table des matières** donne les titres, mais elle ignore parfois des fichiers, ou
  au contraire pointe plusieurs entrées vers des ancres d'un même fichier.

On suit donc le spine pour l'ordre, et l'on va chercher dans la table des matières le
titre de chaque fichier, avec repli sur le premier titre trouvé dans le document.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

from ..document import Chapter, Document, clean_title

# Fichiers de service, à écarter : ils n'appartiennent pas au texte de l'ouvrage.
SKIP_PATTERNS = re.compile(
    r"cover|title[-_]?page|copyright|colophon|nav\b|toc\b|contents|index|acknowledg",
    re.I,
)
# En deçà, un document du spine est une page de garde ou une image pleine page.
MIN_CHAPTER_CHARS = 200
BLOCK_TAGS = ("p", "div", "blockquote", "li")
HEADING_TAGS = ("h1", "h2", "h3", "title")


def _toc_titles(book) -> dict[str, str]:
    """Associe chaque fichier de la table des matières à son intitulé.

    La forme de `book.toc` varie selon les producteurs d'EPUB : liste de liens, mais
    aussi bien lien unique, sections imbriquées, ou couples (section, enfants). Le
    parcours doit accepter tout cela plutôt que de présumer d'une structure.
    """
    titles: dict[str, str] = {}

    def walk(entries) -> None:
        if entries is None:
            return
        if not isinstance(entries, (list, tuple)):
            entries = [entries]
        for entry in entries:
            if isinstance(entry, (list, tuple)):
                walk(entry)
                continue
            href = getattr(entry, "href", None)
            title = getattr(entry, "title", None)
            if href and title:
                # Les ancres (« ch01.xhtml#s2 ») désignent le même fichier.
                titles.setdefault(href.split("#")[0], title)
            # Une section porte ses entrées dans un attribut, pas dans l'itération.
            walk(getattr(entry, "subitems", None))

    walk(book.toc)
    return titles


def _paragraphs(soup: BeautifulSoup) -> list[str]:
    """Extrait les paragraphes, en préférant le balisage à un découpage du texte brut."""
    blocks: list[str] = []
    for element in soup.find_all(BLOCK_TAGS):
        # Un conteneur qui en contient d'autres serait compté deux fois.
        if element.find(BLOCK_TAGS):
            continue
        text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
        if text:
            blocks.append(text)

    if blocks:
        return blocks
    # Balisage pauvre : on se rabat sur les lignes vides du texte brut.
    raw = soup.get_text("\n", strip=True)
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n{2,}", raw) if p.strip()]


def _document_title(soup: BeautifulSoup) -> str:
    for tag in HEADING_TAGS:
        found = soup.find(tag)
        if found:
            text = found.get_text(" ", strip=True)
            if text:
                return text
    return ""


def _metadata(book, key: str, default: str = "") -> str:
    try:
        values = book.get_metadata("DC", key)
    except Exception:
        return default
    return values[0][0] if values else default


def ingest(path: Path, title: str = "", author: str = "") -> Document:
    """Extrait un document depuis un EPUB."""
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path), options={"ignore_ncx": False})
    toc = _toc_titles(book)

    chapters: list[Chapter] = []
    skipped: list[str] = []

    for spine_id, _ in book.spine:
        item = book.get_item_with_id(spine_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        name = item.get_name()
        soup = BeautifulSoup(item.get_content(), "html.parser")
        paragraphs = _paragraphs(soup)
        length = sum(len(p) for p in paragraphs)

        # Un fichier de service reste écartable même s'il est bavard — une page de
        # copyright peut l'être — mais on ne jette jamais un document listé au sommaire.
        listed = name in toc or any(name.endswith(href) for href in toc)
        if (SKIP_PATTERNS.search(name) and not listed) or length < MIN_CHAPTER_CHARS:
            skipped.append(f"{name} ({length} car.)")
            continue

        heading = toc.get(name) or next(
            (t for href, t in toc.items() if name.endswith(href)), ""
        ) or _document_title(soup)

        chapters.append(
            Chapter(
                number=len(chapters) + 1,
                title=clean_title(heading) or f"Chapitre {len(chapters) + 1}",
                paragraphs=paragraphs,
            )
        )

    return Document(
        title=title or _metadata(book, "title") or path.stem,
        author=author or _metadata(book, "creator", "Inconnu"),
        year=(_metadata(book, "date") or None),
        language=_metadata(book, "language", "fr")[:2] or "fr",
        chapters=chapters,
        source=path,
        # Le texte d'un EPUB est du texte, pas une reconnaissance de caractères : il n'y
        # a rien à relire avant de le faire lire.
        needs_review=False,
        notes={"skipped": skipped},
    )
