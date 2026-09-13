"""Tests du repérage des personas : le modèle propose, les garde-fous trient, rien ne
s'écrit sans un geste humain."""

from __future__ import annotations

import json

from voxlibris.llm import LLM, LLMConfig, LLMError
from voxlibris.personas import (
    Block,
    Persona,
    Report,
    discover,
    paragraphs_of,
    pending,
    readers,
)

CHAPITRE = "\n\n".join(
    [
        "Le facteur a apporté une lettre.",  # 0
        "Ma Lulu,",  # 1
        "Depuis mon départ, nous n'avons pas arrêté de bouger.",  # 2
        "Mille bises.",  # 3
        "Charles",  # 4
        "Ce n'était pas une lettre bien longue.",  # 5
        "— Il va bien ! ai-je dit à maman.",  # 6
    ]
)


class FakeLLM(LLM):
    """Un modèle qui répond ce qu'on lui a dit de répondre, et note ce qu'on lui demande."""

    def __init__(self, answer: object):
        super().__init__(LLMConfig(base_url="http://exemple.invalide/v1"))
        self.answer = answer
        self.prompts: list[str] = []

    def ask_json(self, prompt: str, system: str = "") -> object:
        self.prompts.append(prompt)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class TestReperage:
    def test_un_bloc_valide_est_propose_avec_ses_bornes_en_texte(self):
        llm = FakeLLM(
            {
                "personas": [{"nom": "Charles", "qui": "le frère de Lulu"}],
                "blocs": [{"persona": "Charles", "de": 1, "a": 4}],
            }
        )
        report = discover({"ch01.md": CHAPITRE}, llm=llm)
        assert [p.name for p in report.personas] == ["Charles"]
        assert report.blocks == [Block("ch01.md", "Charles", 1, 4, "Ma Lulu,", "Charles")]
        assert report.rejected == [] and report.chapters == 1
        # Les paragraphes sont numérotés pour le modèle, qui n'a qu'à recopier.
        assert "[1] Ma Lulu," in llm.prompts[0] and "[4] Charles" in llm.prompts[0]

    def test_les_garde_fous_ecartent_ce_qui_ne_tient_pas(self):
        llm = FakeLLM(
            {
                "personas": [{"nom": "Charles"}, {"nom": "narrateur"}, {"nom": ""}],
                "blocs": [
                    {"persona": "Charles", "de": 1, "a": 4},
                    {"persona": "Charles", "de": 3, "a": 5},  # chevauche
                    {"persona": "Jules", "de": 6, "a": 6},  # inconnu
                    {"persona": "Charles", "de": 5, "a": 40},  # hors du chapitre
                    {"persona": "Charles", "de": "un", "a": 2},  # illisible
                ],
            }
        )
        report = discover({"ch01.md": CHAPITRE}, llm=llm)
        assert [p.name for p in report.personas] == ["Charles"]
        assert [(b.first, b.last) for b in report.blocks] == [(1, 4)]
        assert len(report.rejected) == 4

    def test_un_persona_connu_garde_son_nom_et_ce_qui_est_fait_n_est_pas_repropose(self):
        marked = CHAPITRE.replace("Ma Lulu,", "@charles\n\nMa Lulu,").replace(
            "Charles\n\nCe", "Charles\n\n@\n\nCe"
        )
        llm = FakeLLM(
            {
                "personas": [{"nom": "charles", "qui": "le frère"}],
                "blocs": [{"persona": "charles", "de": 2, "a": 5}],
            }
        )
        report = discover({"ch01.md": marked}, known=["Charles"], llm=llm)
        assert "Personas déjà connus : Charles" in llm.prompts[0]
        # Déjà connu, sous une autre casse : cité au rapport sous son nom connu.
        assert report.personas == [Persona("Charles", "le frère")]
        # Déjà attribué dans le texte : au rapport, mais plus rien à proposer.
        assert [b.persona for b in report.blocks] == ["Charles"] and report.rejected == []
        assert pending(report.blocks, marked) == []
        # Les marqueurs ne sont pas montrés au modèle, mais ils comptent dans les numéros,
        # qui restent ceux du texte tel qu'il est.
        assert "[2] Ma Lulu," in llm.prompts[0] and "@charles" not in llm.prompts[0]

    def test_une_panne_du_modele_laisse_un_rapport_avec_son_motif(self):
        report = discover({"ch01.md": CHAPITRE}, llm=FakeLLM(LLMError("injoignable")))
        assert report.error == "injoignable" and report.blocks == []

    def test_le_rapport_fait_l_aller_retour(self):
        report = Report(
            personas=[],
            blocks=[Block("ch01.md", "Charles", 1, 4, "Ma Lulu,", "Charles")],
            chapters=1,
        )
        again = Report.from_json(report.to_json())
        assert again == report
        assert json.loads(report.to_json())["blocks"][0]["persona"] == "Charles"


class TestEnAttente:
    def test_un_bloc_reste_propose_tant_qu_il_n_est_pas_attribue(self):
        block = Block("ch01.md", "Charles", 1, 4, "Ma Lulu,", "Charles")
        assert pending([block], CHAPITRE) == [(block, 0)]
        done = CHAPITRE.replace("Ma Lulu,", "@Charles\n\nMa Lulu,").replace(
            "Charles\n\nCe", "Charles\n\n@\n\nCe"
        )
        assert pending([block], done) == []
        # Le texte a été réécrit : la borne n'existe plus, la proposition tombe.
        assert pending([block], CHAPITRE.replace("Ma Lulu,", "Ma Lucienne,")) == []

    def test_deux_lettres_qui_s_ouvrent_pareil_ne_se_confondent_pas(self):
        """La première lettre est attribuée ; la seconde, même salutation, reste à faire,
        et l'éditeur reçoit son rang pour viser la bonne."""
        first = CHAPITRE.replace("Ma Lulu,", "@Charles\n\nMa Lulu,").replace(
            "Charles\n\nCe", "Charles\n\n@\n\nCe"
        )
        text = first + "\n\nPlus tard.\n\nMa Lulu,\n\nTout va bien.\n\nCharles\n\nFin."
        paragraphs = paragraphs_of(text)
        second = paragraphs.index("Ma Lulu,", 3)
        block = Block("ch01.md", "Charles", second, second + 2, "Ma Lulu,", "Charles")
        assert pending([block], text) == [(block, 1)]

        text = "Récit.\n\n@Charles\n\nLettre.\n\n@\n\nSuite."
        assert readers(paragraphs_of(text)) == ["narrateur", "", "Charles", "", "narrateur"]
