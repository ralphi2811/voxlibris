"""Fabriques de livres de test.

Les EPUB et PDF utilisés par les tests sont construits ici, à la volée. Aucun fichier
n'est téléchargé et aucune œuvre n'est embarquée dans le dépôt : les tests restent
hermétiques, reproductibles, et sans la moindre question de droits.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Texte inventé pour les besoins des tests, assez long pour dépasser les seuils qui
# écartent les pages de garde.
LOREM = (
    "Le gardien du phare notait chaque soir la couleur exacte du ciel. "
    "Il tenait ce registre depuis vingt ans, sans en avoir jamais manqué un seul. "
    "Personne ne le lui avait demandé, et personne ne le lui avait interdit. "
)


def _chapter_html(title: str, paragraphs: int = 3) -> str:
    body = "\n".join(f"<p>{LOREM * 2}</p>" for _ in range(paragraphs))
    return f"<html><head><title>{title}</title></head><body><h1>{title}</h1>{body}</body></html>"


@pytest.fixture
def make_epub(tmp_path: Path):
    """Fabrique un EPUB minimal mais valide, avec sommaire et pages de service."""

    def build(
        titles: list[str],
        name: str = "livre.epub",
        with_cover: bool = True,
        in_toc: bool = True,
    ) -> Path:
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("test-voxlibris")
        book.set_title("Le registre du gardien")
        book.set_language("fr")
        book.add_author("Anonyme")

        spine: list = []
        toc: list = []

        if with_cover:
            cover = epub.EpubHtml(title="Couverture", file_name="cover.xhtml", lang="fr")
            cover.content = "<html><body><h1>Le registre du gardien</h1></body></html>"
            book.add_item(cover)
            spine.append(cover)

        for index, title in enumerate(titles, start=1):
            item = epub.EpubHtml(title=title, file_name=f"ch{index:02d}.xhtml", lang="fr")
            item.content = _chapter_html(title)
            book.add_item(item)
            spine.append(item)
            if in_toc:
                toc.append(item)

        book.toc = tuple(toc)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = spine

        path = tmp_path / name
        epub.write_epub(str(path), book)
        return path

    return build


@pytest.fixture
def make_pdf(tmp_path: Path):
    """Fabrique un PDF à texte natif, avec ou sans signets de chapitres."""

    def build(titles: list[str], name: str = "livre.pdf", outline: bool = True) -> Path:
        import pymupdf

        doc = pymupdf.open()
        starts: list[int] = []
        for title in titles:
            starts.append(len(doc))
            for part in range(2):
                page = doc.new_page(width=420, height=595)
                if part == 0:
                    page.insert_text((72, 90), title, fontsize=20)
                page.insert_textbox(
                    pymupdf.Rect(72, 130, 348, 520), LOREM * 6, fontsize=11
                )

        if outline:
            doc.set_toc(
                [[1, t, s + 1] for t, s in zip(titles, starts, strict=True)]
            )

        path = tmp_path / name
        doc.save(str(path))
        return path

    return build
