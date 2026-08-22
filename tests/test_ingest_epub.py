"""Tests de l'ingestion EPUB.

C'est le format qui porte la promesse d'automatisation : si l'EPUB passe sans
intervention, tout le reste de la chaîne suit. D'où l'attention portée ici au
chapitrage et à l'exclusion des pages de service.
"""

from __future__ import annotations

from voxlibris.ingest import epub as E

TITRES = ["Le départ", "La traversée", "L'arrivée"]


class TestChapitrage:
    def test_un_chapitre_par_document(self, make_epub):
        doc = E.ingest(make_epub(TITRES))
        assert [c.title for c in doc.chapters] == TITRES

    def test_numerotation_contigue(self, make_epub):
        doc = E.ingest(make_epub(TITRES))
        assert [c.number for c in doc.chapters] == [1, 2, 3]

    def test_titres_deduits_du_document_sans_sommaire(self, make_epub):
        # Sans table des matières, le premier titre du document fait foi.
        doc = E.ingest(make_epub(TITRES, in_toc=False))
        assert [c.title for c in doc.chapters] == TITRES

    def test_ordre_de_lecture_respecte(self, make_epub):
        doc = E.ingest(make_epub(["Zeta", "Alpha", "Mu"]))
        # L'ordre vient du spine, surtout pas d'un tri alphabétique.
        assert [c.title for c in doc.chapters] == ["Zeta", "Alpha", "Mu"]


class TestPagesDeService:
    def test_couverture_ecartee(self, make_epub):
        doc = E.ingest(make_epub(TITRES, with_cover=True))
        assert len(doc.chapters) == len(TITRES)
        assert all("registre du gardien" not in c.title.lower() for c in doc.chapters)

    def test_exclusion_tracee(self, make_epub):
        doc = E.ingest(make_epub(TITRES, with_cover=True))
        assert any("cover" in note for note in doc.notes["skipped"])


class TestContenu:
    def test_paragraphes_conserves(self, make_epub):
        doc = E.ingest(make_epub(TITRES))
        assert all(len(c.paragraphs) >= 3 for c in doc.chapters)

    def test_texte_non_vide(self, make_epub):
        doc = E.ingest(make_epub(TITRES))
        assert doc.word_count > 100

    def test_titre_du_chapitre_absent_du_corps(self, make_epub):
        # Le <h1> sert de titre : il ne doit pas être lu une seconde fois.
        doc = E.ingest(make_epub(TITRES))
        assert doc.chapters[0].paragraphs[0] != TITRES[0]


class TestMetadonnees:
    def test_titre_et_auteur_lus_dans_lepub(self, make_epub):
        doc = E.ingest(make_epub(TITRES))
        assert doc.title == "Le registre du gardien"
        assert doc.author == "Anonyme"

    def test_aucune_relecture_requise(self, make_epub):
        # Un EPUB n'est pas une reconnaissance de caractères : rien à relire.
        assert E.ingest(make_epub(TITRES)).needs_review is False

    def test_surcharge_des_metadonnees(self, make_epub):
        doc = E.ingest(make_epub(TITRES), title="Autre titre", author="Autre auteur")
        assert (doc.title, doc.author) == ("Autre titre", "Autre auteur")
