"""Pages d'origine d'un EPUB paginé, servies en regard du texte.

Le livre de test est celui de `test_ingest_epub_fixe` : des pages pdf2htmlEX assemblées à
la main, avec un script et un en-tête de rafraîchissement comme dans les vrais.
"""

from __future__ import annotations

import pytest
from test_ingest_epub_fixe import PAGE_H, page, write_epub

from voxlibris import pages
from voxlibris.project import Project

SCRIPTED = (
    '<script src="pdf2fl.js" type="text/javascript"> </script>'
    '<meta content="text/html; charset=UTF-8" http-equiv="Refresh"/>'
)


@pytest.fixture
def book(tmp_path):
    first = page([("x0", 600, "h0", "<span>Première page du registre.</span>")])
    second = page([("x0", 600, "h0", "<span>Seconde page du registre.</span>")], folio="2")
    # Un script et une balise Refresh dans la seconde page, comme pdf2htmlEX les sème.
    second = second.replace("<head>", "<head>" + SCRIPTED, 1)
    fonts = {"fonts/a.ttf": b"\x00\x01\x00\x00police", "pdf2fl.js": b"alert(1)"}
    return write_epub(tmp_path / "registre.epub", [first, second], [(1, "1. Le début")], fonts)


class TestPages:
    def test_ordre_du_livre(self, book):
        found = pages.Pages(book)
        assert len(found) == 2
        assert found.names == ["OPS/pageNum-1.html", "OPS/pageNum-2.html"]

    def test_la_page_est_servie_sans_script(self, book):
        html = pages.Pages(book).page(1, "/projects/registre/source")
        assert "<script" not in html
        assert "alert" not in html
        assert "Refresh" not in html
        assert '<base href="/projects/registre/source/OPS/">' in html
        assert "Seconde page du registre." in html

    def test_dimensions(self, book):
        assert pages.size(pages.Pages(book).raw(0)) == (1000, PAGE_H)
        assert pages.size("<html><head></head></html>") is None
        # Sans balise viewport, le corps de page dit sa taille.
        body = '<body style="position:absolute;width:1414px;height:2252px;">'
        assert pages.size(f"<html><head></head>{body}</body></html>") == (1414, 2252)

    def test_les_ressources_d_affichage_seulement(self, book):
        found = pages.Pages(book)
        data, media = found.asset("OPS/fonts/a.ttf")
        assert media == "font/ttf" and data.endswith(b"police")
        # Ni les scripts, ni les pages, ni le manifeste, ni hors du livre.
        assert found.asset("OPS/pdf2fl.js") is None
        assert found.asset("OPS/pageNum-1.html") is None
        assert found.asset("OPS/content.opf") is None
        assert found.asset("../etc/passwd.png") is None
        assert found.asset("OPS/fonts/absente.ttf") is None

    def test_genre_de_source(self, tmp_path):
        project = Project(root=tmp_path, source="livre.pdf")
        assert pages.kind(project) == "pdf"
        project = Project(root=tmp_path, source="livre.epub")
        assert pages.kind(project) is None
        project.notes["layout"] = pages.LAYOUT
        assert pages.kind(project) == "epub"
        assert pages.kind(Project(root=tmp_path)) is None
