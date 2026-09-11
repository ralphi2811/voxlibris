"""Métadonnées lues au dépôt, pour préremplir le formulaire."""

from __future__ import annotations

from voxlibris.ingest.metadata import cover, peek, search_cover


class TestPeek:
    def test_epub(self, make_epub):
        found = peek(make_epub(["Le départ"]))
        assert found["title"] == "Le registre du gardien"
        assert found["language"] == "fr"

    def test_pdf(self, tmp_path):
        import pymupdf

        path = tmp_path / "livre.pdf"
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "Bonjour")
        doc.set_metadata(
            {"title": "Le phare", "author": "Anonyme", "creationDate": "D:20180116160000"}
        )
        doc.save(path)
        assert peek(path) == {"title": "Le phare", "author": "Anonyme", "year": "2018"}

    def test_texte_gutenberg(self, tmp_path):
        path = tmp_path / "livre.txt"
        path.write_text(
            "The Project Gutenberg eBook\n\nTitle: Le Horla\n\nAuthor: Guy de Maupassant\n\n"
            "Language: French\n\nCHAPITRE I\n",
            encoding="utf-8",
        )
        assert peek(path) == {"title": "Le Horla", "author": "Guy de Maupassant", "language": "fr"}

    def test_markdown(self, tmp_path):
        path = tmp_path / "livre.md"
        path.write_text("# Le registre\n\nTexte.\n", encoding="utf-8")
        assert peek(path) == {"title": "Le registre"}

    def test_auteur_remis_a_l_endroit(self, tmp_path):
        path = tmp_path / "livre.txt"
        path.write_text("Title: Le livre magique\nAuthor: Black, Holly\n", encoding="utf-8")
        assert peek(path)["author"] == "Holly Black"

    def test_rien_a_dire(self, tmp_path):
        path = tmp_path / "livre.txt"
        path.write_text("Texte sans en-tête.\n", encoding="utf-8")
        assert peek(path) == {}
        broken = tmp_path / "cassé.epub"
        broken.write_bytes(b"pas un zip")
        assert peek(broken) == {}


def media(href: str) -> str:
    return "image/jpeg" if href.endswith(".jpg") else "application/xhtml+xml"


def epub_with(tmp_path, opf_extra: str, files: dict[str, bytes], name="livre.epub"):
    import zipfile

    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            '<rootfile full-path="OPS/content.opf"/></rootfiles></container>',
        )
        zf.writestr(
            "OPS/content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf"><metadata>'
            + opf_extra
            + "</metadata><manifest>"
            + "".join(
                f'<item id="{i}" href="{href}" media-type="{media(href)}"/>'
                for i, href in enumerate(files)
            )
            + '</manifest><spine><itemref idref="0"/></spine></package>',
        )
        for href, data in files.items():
            zf.writestr(f"OPS/{href}", data)
    return path


class TestCouverture:
    def test_designee_par_le_manifeste(self, tmp_path):
        path = epub_with(
            tmp_path,
            '<meta name="cover" content="1"/>',
            {"p1.html": b"<html/>", "images/c.jpg": b"JPEG-COUV"},
        )
        assert cover(path) == b"JPEG-COUV"

    def test_designee_par_son_nom_facon_calibre(self, tmp_path):
        path = epub_with(
            tmp_path,
            '<meta name="cover" content="c.jpg"/>',
            {"p1.html": b"<html/>", "images/c.jpg": b"JPEG-COUV"},
        )
        assert cover(path) == b"JPEG-COUV"

    def test_sinon_l_image_de_la_premiere_page(self, tmp_path):
        page = b'<html><body><img src="images/img-1-1.jpg"/></body></html>'
        path = epub_with(tmp_path, "", {"p1.html": page, "images/img-1-1.jpg": b"PAGE-UNE"})
        assert cover(path) == b"PAGE-UNE"

    def test_rien(self, tmp_path):
        path = epub_with(tmp_path, "", {"p1.html": b"<html/>"})
        assert cover(path) is None
        broken = tmp_path / "cassé.epub"
        broken.write_bytes(b"pas un zip")
        assert cover(broken) is None

    def test_recherche_en_ligne(self):
        calls = []

        def fetch(url):
            calls.append(url)
            if "search.json" in url:
                return b'{"docs": [{"title": "sans image"}, {"cover_i": 42}]}'
            return b"J" * 5000

        assert search_cover("Le Horla", "Maupassant", fetch=fetch) == b"J" * 5000
        assert "title=Le+Horla" in calls[0] and "author=Maupassant" in calls[0]
        assert calls[1].endswith("/b/id/42-L.jpg?default=false")

    def test_recherche_sans_resultat(self):
        assert search_cover("Le Horla", fetch=lambda url: b'{"docs": []}') is None
        assert search_cover("Le Horla", fetch=lambda url: b"pas du json") is None
        assert search_cover("   ") is None
