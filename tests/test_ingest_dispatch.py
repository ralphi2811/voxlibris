"""Tests de la détection de format et des lecteurs PDF natif et texte brut."""

from __future__ import annotations

import pytest

from voxlibris import ingest
from voxlibris.ingest import Kind
from voxlibris.ingest import pdf_text as P
from voxlibris.ingest import plain as T

TITRES = ["Le départ", "La traversée", "L'arrivée"]


class TestDetection:
    def test_epub(self, make_epub):
        assert ingest.detect_kind(make_epub(TITRES)) is Kind.EPUB

    def test_pdf_natif(self, make_pdf):
        assert ingest.detect_kind(make_pdf(TITRES)) is Kind.PDF_TEXT

    def test_texte_brut(self, tmp_path):
        path = tmp_path / "livre.txt"
        path.write_text("Du texte.", encoding="utf-8")
        assert ingest.detect_kind(path) is Kind.PLAIN

    def test_markdown(self, tmp_path):
        path = tmp_path / "livre.md"
        path.write_text("# Titre\n\nDu texte.", encoding="utf-8")
        assert ingest.detect_kind(path) is Kind.PLAIN

    def test_pdf_sans_texte(self, tmp_path):
        import pymupdf

        doc = pymupdf.open()
        for _ in range(3):
            doc.new_page(width=300, height=400)
        path = tmp_path / "scan.pdf"
        doc.save(str(path))
        assert ingest.detect_kind(path) is Kind.PDF_IMAGE

    def test_format_inconnu(self, tmp_path):
        path = tmp_path / "livre.docx"
        path.write_bytes(b"peu importe")
        with pytest.raises(ValueError, match="non pris en charge"):
            ingest.detect_kind(path)


class TestAiguillage:
    def test_epub_traverse_laiguillage(self, make_epub):
        doc = ingest.ingest(make_epub(TITRES))
        assert len(doc.chapters) == 3
        assert doc.notes["kind"] == "epub"

    def test_pdf_sans_texte_refuse_explicitement(self, tmp_path):
        import pymupdf

        doc = pymupdf.open()
        doc.new_page()
        path = tmp_path / "scan.pdf"
        doc.save(str(path))
        with pytest.raises(NotImplementedError, match="océriser"):
            ingest.ingest(path)

    def test_fichier_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest.ingest(tmp_path / "fantome.epub")


class TestPdfNatif:
    def test_chapitres_depuis_les_signets(self, make_pdf):
        doc = P.ingest(make_pdf(TITRES, outline=True))
        assert [c.title for c in doc.chapters] == TITRES
        assert doc.notes["chapitrage"] == "signets"

    def test_repli_sur_les_tailles_de_police(self, make_pdf):
        doc = P.ingest(make_pdf(TITRES, outline=False))
        assert doc.notes["chapitrage"] == "tailles de police"
        assert len(doc.chapters) == len(TITRES)

    def test_aucune_relecture_requise(self, make_pdf):
        assert P.ingest(make_pdf(TITRES)).needs_review is False

    def test_titre_absent_du_corps(self, make_pdf):
        doc = P.ingest(make_pdf(TITRES))
        assert not doc.chapters[0].paragraphs[0].startswith(TITRES[0])


class TestTexteBrut:
    def test_chapitres_par_titres_markdown(self, tmp_path):
        path = tmp_path / "livre.md"
        path.write_text(
            "# Le départ\n\nUn paragraphe.\n\n# La traversée\n\nUn autre.\n", encoding="utf-8"
        )
        doc = T.ingest(path)
        assert [c.title for c in doc.chapters] == ["Le départ", "La traversée"]
        assert doc.notes["chapitrage"] == "titres Markdown"

    def test_chapitres_par_lignes_de_titre(self, tmp_path):
        path = tmp_path / "livre.txt"
        path.write_text(
            "CHAPITRE PREMIER\n\nUn paragraphe.\n\nCHAPITRE DEUX\n\nUn autre.\n",
            encoding="utf-8",
        )
        doc = T.ingest(path)
        assert len(doc.chapters) == 2

    def test_texte_sans_repere_forme_un_chapitre(self, tmp_path):
        path = tmp_path / "livre.txt"
        path.write_text("Un paragraphe.\n\nUn autre paragraphe.\n", encoding="utf-8")
        doc = T.ingest(path)
        assert len(doc.chapters) == 1
        assert len(doc.chapters[0].paragraphs) == 2

    def test_lignes_recollees_en_paragraphes(self, tmp_path):
        # Un texte replié à 70 colonnes ne doit pas donner un paragraphe par ligne.
        path = tmp_path / "livre.txt"
        path.write_text("Une phrase coupée\nsur deux lignes.\n\nUne autre.\n", encoding="utf-8")
        doc = T.ingest(path)
        assert doc.chapters[0].paragraphs[0] == "Une phrase coupée sur deux lignes."


class TestGutenberg:
    """L'enveloppe de Project Gutenberg tombe d'elle-même, à l'ingestion."""

    def _doc(self):
        from voxlibris.document import Chapter, Document

        return Document(
            title="Essai",
            author="A.",
            chapters=[
                Chapter(1, "Notice", ["Title: Essai", "This eBook is for the use of anyone.",
                                      "*** START OF THE PROJECT GUTENBERG EBOOK ESSAI ***",
                                      "Il était une fois."]),
                Chapter(2, "Suite", ["La suite du récit.", "Fin.",
                                     "*** END OF THE PROJECT GUTENBERG EBOOK ESSAI ***",
                                     "Section 1. General Terms of Use"]),
                Chapter(3, "Licence", ["Blabla juridique."]),
            ],
        )

    def test_preambule_et_licence_retires(self):
        from voxlibris.document import strip_gutenberg

        doc = self._doc()
        removed = strip_gutenberg(doc)
        assert [c.paragraphs for c in doc.chapters] == [
            ["Il était une fois."],
            ["La suite du récit.", "Fin."],
        ]
        assert [c.number for c in doc.chapters] == [1, 2]
        assert removed == 6

    def test_sans_enveloppe_rien_ne_bouge(self):
        from voxlibris.document import Chapter, Document, strip_gutenberg

        chapters = [Chapter(1, "Un", ["Bonjour.", "Au revoir."])]
        doc = Document(title="T", author="A", chapters=chapters)
        assert strip_gutenberg(doc) == 0
        assert doc.chapters[0].paragraphs == ["Bonjour.", "Au revoir."]
