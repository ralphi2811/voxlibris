"""Tests de la reconstruction des paragraphes d'un scan, à partir des lignes."""

from __future__ import annotations

from voxlibris.ingest.layout import Layout
from voxlibris.ingest.pdf_scan import Line, build_paragraphs, is_garbage

LAYOUT = Layout(600, 40, 560, 10, 8, 12, 40)


def lines(*rows: tuple[str, float, float]) -> list[Line]:
    return [Line(text, 60 + 14 * i, x, 10, 1, width) for i, (text, x, width) in enumerate(rows)]


class TestParagraphes:
    def test_un_alinea_ouvre_un_paragraphe(self):
        found = build_paragraphs(
            lines(("Première.", 40, 300), ("    Suite indentée.", 52, 300), ("Fin.", 40, 300))
        )
        assert found == ["Première.", "Suite indentée. Fin."]

    def test_une_ligne_courte_close_ferme_le_paragraphe(self):
        found = build_paragraphs(
            lines(
                ("Une ligne pleine qui continue", 40, 300),
                ("et s'achève ici.", 40, 120),
                ("Nouveau départ, sans alinéa,", 40, 300),
                ("jusqu'au bout.", 40, 300),
            )
        )
        assert found == [
            "Une ligne pleine qui continue et s'achève ici.",
            "Nouveau départ, sans alinéa, jusqu'au bout.",
        ]

    def test_une_ligne_courte_sans_ponctuation_ne_ferme_rien(self):
        found = build_paragraphs(
            lines(
                ("Une ligne pleine, puis un mot", 40, 300),
                ("seul", 40, 60),
                ("avant la suite.", 40, 300),
            )
        )
        assert found == ["Une ligne pleine, puis un mot seul avant la suite."]

    def test_sans_largeur_connue_la_regle_ne_joue_pas(self):
        found = build_paragraphs([Line("Courte.", 60, 40, 10, 1), Line("Suite.", 74, 40, 10, 1)])
        assert found == ["Courte. Suite."]


class TestRebut:
    def test_les_guillemets_typographiques_ne_font_pas_un_rebut(self):
        assert not is_garbage("— “Merci Auguste.’", 10, LAYOUT)

    def test_un_fragment_dillustration_reste_un_rebut(self):
        assert is_garbage("L% Æ", 10, LAYOUT)
