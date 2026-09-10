"""Les garde-fous, éprouvés sans modèle.

Ce sont eux qui décident de ce qui atteint l'écran d'un relecteur. Un modèle réel n'a
rien à faire ici : il rendrait ces tests lents, dépendants d'un service, et surtout non
reproductibles — alors que la question posée, elle, est parfaitement déterministe.
La qualité du modèle se mesure ailleurs, avec « voxlibris bench-proofread ».
"""

from __future__ import annotations

import json

import pytest
from spellchecker import SpellChecker

from voxlibris.llm import LLM, THINK, LLMConfig
from voxlibris.proofread import apply, check, distance, suggest


class FakeLLM(LLM):
    """Un modèle qui répond ce qu'on lui a dit de répondre."""

    def __init__(self, answers: dict[str, object]):
        super().__init__(LLMConfig(base_url="http://exemple.invalide/v1"))
        self.answers = answers
        self.prompts: list[str] = []

    def ask_json(self, prompt: str, system: str = "") -> object:
        self.prompts.append(prompt)
        entries = json.loads(prompt[prompt.index("[") : prompt.rindex("]") + 1])
        return [
            {"id": entry["id"], "correction": self.answers.get(entry["forme"])} for entry in entries
        ]


@pytest.fixture
def spell():
    return SpellChecker(language="fr")


def test_distance():
    assert distance("il", "il") == 0
    assert distance("1l", "il") == 1
    assert distance("pardessus", "par-dessus") == 1


@pytest.mark.parametrize(
    "word,proposal,expected",
    [
        ("1l", "il", "il"),
        ("connaîftront", "connaîtront", "connaîtront"),
        ("vingtquatre", "vingt-quatre", "vingt-quatre"),
    ],
)
def test_corrections_plausibles_acceptees(word, proposal, expected, spell):
    assert check(word, proposal, spell, set()) == (expected, "")


def test_une_reecriture_est_refusee(spell):
    """Une coquille reste proche du mot ; au-delà, c'est une invention."""
    _, cause = check("comm", "cependant", spell, set())
    assert "distance" in cause


def test_la_ponctuation_est_intouchable(spell):
    """La ponctuation est calibrée pour la synthèse : le modèle n'y touche pas."""
    _, cause = check("dit", "dit.", spell, set())
    assert cause == "ponctuation introduite"


def test_un_mot_inexistant_est_refuse(spell):
    _, cause = check("chaïirs", "chairz", spell, set())
    assert "hors dictionnaire" in cause


def test_une_phrase_est_refusee(spell):
    _, cause = check("Etils", "Et ils allaient", spell, set())
    assert "3 mots" in cause


def test_le_silence_du_modele_ne_produit_rien(spell):
    assert check("Mâdâme", None, spell, set()) == ("", "")
    assert check("Mâdâme", "Mâdâme", spell, set()) == ("", "")


def test_une_forme_recurrente_nest_pas_soumise():
    """Ce qui revient est voulu : le parler d'un personnage, pas une coquille."""
    texte = " ".join(["Bien le bonjour, vot' serviteur."] * 6)
    llm = FakeLLM({"vot'": "votre"})
    report = suggest({"ch01.md": texte}, llm=llm, recurrence_limit=4)

    assert not report.suggestions
    assert report.skipped == ["vot'"]
    assert not llm.prompts  # le modèle n'a même pas été interrogé


def test_une_coquille_isolee_est_proposee():
    texte = "Le navire connaîftront la tempête au large de Maurice."
    report = suggest({"ch01.md": texte}, llm=FakeLLM({"connaîftront": "connaîtront"}))

    assert [(s.word, s.replacement) for s in report.suggestions] == [
        ("connaîftront", "connaîtront")
    ]
    assert report.suggestions[0].context.count("⟦") == 1


def test_une_proposition_invalide_est_ecartee_et_tracee():
    texte = "Le navire connaîftront la tempête au large de Maurice."
    report = suggest({"ch01.md": texte}, llm=FakeLLM({"connaîftront": "sombrèrent"}))

    assert not report.suggestions
    assert report.rejected and "distance" in report.rejected[0].cause


def test_desactive_par_configuration():
    report = suggest({"ch01.md": "peu importe"}, config=LLMConfig(enabled=False))
    assert not report.suggestions
    assert "désactivée" in report.error


def test_une_panne_du_modele_nempeche_pas_de_relire(monkeypatch):
    """Sans modèle, la relecture manuelle reste possible : le rapport dit pourquoi."""
    texte = "Le navire connaîftront la tempête au large de Maurice."
    llm = FakeLLM({})
    monkeypatch.setattr(
        llm,
        "ask_json",
        lambda *a, **k: (_ for _ in ()).throw(
            __import__("voxlibris.llm", fromlist=["LLMError"]).LLMError("injoignable")
        ),
    )
    report = suggest({"ch01.md": texte}, llm=llm)
    assert not report.suggestions
    assert "injoignable" in report.error


def test_apply_ne_corrige_pas_a_laveugle():
    texte = "Le navire connaîftront la tempête."
    report = suggest({"ch01.md": texte}, llm=FakeLLM({"connaîftront": "connaîtront"}))
    assert apply(texte, report.suggestions) == "Le navire connaîtront la tempête."
    # Le relecteur est passé avant : la position ne désigne plus la même forme.
    assert apply("Tout autre chose ici.", report.suggestions) == "Tout autre chose ici."


def test_le_monologue_des_modeles_est_retire():
    assert THINK.sub("", "<think>hmm…</think>[1]").strip() == "[1]"


def test_configuration_lue_dans_lenvironnement():
    config = LLMConfig.from_env(
        {"VOXLIBRIS_LLM_MODEL": "mistral", "VOXLIBRIS_LLM_BASE_URL": "http://ailleurs:8000/v1"}
    )
    assert config.model == "mistral"
    assert config.endpoint == "http://ailleurs:8000/v1/chat/completions"
    assert config.enabled


def test_configuration_desactivable():
    assert not LLMConfig.from_env({"VOXLIBRIS_LLM_ENABLED": "0"}).enabled
    assert not LLMConfig.from_env({"VOXLIBRIS_LLM_ENABLED": "non"}).enabled
