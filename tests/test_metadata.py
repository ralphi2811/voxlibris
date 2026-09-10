"""Métadonnées lues au dépôt, pour préremplir le formulaire."""

from __future__ import annotations

from voxlibris.ingest.metadata import peek


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
