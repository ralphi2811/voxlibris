"""Tests du contrôle qualité.

Ces détecteurs ont demandé trois passes de calibration successives avant d'attraper les
défauts que l'oreille entendait immédiatement : chacun était aveugle à un cas que les
autres ne couvraient pas. Une régression silencieuse ici produirait un livre audio
truffé de babil sans qu'aucune alerte ne se déclenche — d'où ces tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxlibris.tts import quality
from voxlibris.tts.quality import SAMPLE_RATE, QualityProfile


def speech(seconds: float, level: float = 0.2) -> np.ndarray:
    """Signal de niveau audible, tenant lieu de parole."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * level).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), np.float32)


PROFILE = QualityProfile()


class TestTrimEdges:
    def test_retire_les_silences_de_bord(self):
        audio = np.concatenate([silence(1.0), speech(2.0), silence(1.5)])
        trimmed = quality.trim_edges(audio)
        # La marge de garde volontaire empêche une égalité stricte.
        assert 2.0 <= len(trimmed) / SAMPLE_RATE < 2.3

    def test_preserve_un_signal_deja_net(self):
        audio = speech(2.0)
        assert len(quality.trim_edges(audio)) == pytest.approx(len(audio), rel=0.05)

    def test_supporte_le_silence_integral(self):
        audio = silence(1.0)
        assert len(quality.trim_edges(audio)) == len(audio)


class TestTrailingArtifact:
    def test_detecte_un_residu_bref_apres_un_blanc(self):
        audio = np.concatenate([speech(3.0), silence(0.8), speech(0.2)])
        assert quality.trailing_artifact(audio, PROFILE)

    def test_ignore_une_pause_de_ponctuation_suivie_de_parole(self):
        # Un blanc court suivi d'une vraie fin de phrase n'est pas un artefact.
        audio = np.concatenate([speech(3.0), silence(0.3), speech(1.5)])
        assert not quality.trailing_artifact(audio, PROFILE)

    def test_ignore_un_enonce_continu(self):
        assert not quality.trailing_artifact(speech(4.0), PROFILE)


class TestTrailingOverrun:
    """Le défaut que la durée totale ne voit pas : du babil collé à la dernière phrase."""

    TEXTE = "Le vent tomba peu avant l'aube. La mer devint lisse."

    def test_detecte_une_derniere_phrase_qui_deborde(self):
        # « La mer devint lisse. » vaut ~1 s ; ici la dernière plage en fait 2,7.
        audio = np.concatenate([speech(4.0), silence(0.5), speech(2.7)])
        assert quality.trailing_overrun(audio, self.TEXTE, PROFILE)

    def test_accepte_une_derniere_phrase_de_duree_plausible(self):
        audio = np.concatenate([speech(4.0), silence(0.5), speech(1.0)])
        assert not quality.trailing_overrun(audio, self.TEXTE, PROFILE)

    def test_inapplicable_a_une_phrase_unique(self):
        # Sans silence interne, aucun repère ne délimite la dernière plage.
        assert not quality.trailing_overrun(speech(9.0), "Une seule phrase ici.", PROFILE)


class TestInspect:
    def test_plafond_absolu_attrape_les_textes_courts(self):
        """Le trou par lequel une phrase de dix caractères a duré plus de quatre secondes."""
        texte = "La marée."
        assert len(texte) < PROFILE.min_chars  # sous le seuil du contrôle par ratio
        assert quality.inspect(speech(4.5), texte, PROFILE) == "durée absurde"

    def test_accepte_une_duree_plausible(self):
        texte = "Le gardien du phare notait chaque soir la couleur exacte du ciel."
        assert quality.inspect(speech(len(texte) / 18), texte, PROFILE) == ""

    def test_signale_un_enonce_trop_lent(self):
        texte = "Le gardien du phare notait chaque soir la couleur exacte du ciel."
        assert quality.inspect(speech(len(texte) / 8), texte, PROFILE) == "durée absurde"

    def test_signale_un_enonce_trop_rapide(self):
        texte = "Le gardien du phare notait chaque soir la couleur exacte du ciel."
        assert quality.inspect(speech(len(texte) / 40), texte, PROFILE) == "durée"


class TestRender:
    def test_rejoue_puis_retient_la_prise_saine(self):
        texte = "Le gardien du phare notait chaque soir la couleur exacte du ciel."
        bonne = speech(len(texte) / 18)
        tirages = [speech(len(texte) / 4), speech(len(texte) / 5), bonne]

        take = quality.render(lambda _: tirages.pop(0), texte, PROFILE)
        assert take.clean
        assert take.attempts == 3

    def test_abandonne_apres_le_maximum_et_garde_la_moins_mauvaise(self):
        texte = "Le gardien du phare notait chaque soir la couleur exacte du ciel."
        profile = QualityProfile(max_attempts=3)
        durees = [20.0, 8.0, 25.0]  # la deuxième est la plus proche de l'attendu

        take = quality.render(lambda _: speech(durees.pop(0)), texte, profile)
        assert not take.clean
        assert take.attempts == 3
        assert take.duration == pytest.approx(8.0, rel=0.05)


class TestCalibrate:
    def test_mediane_des_debits(self):
        assert quality.calibrate([("x" * 90, 5.0)] * 10) == 18.0

    def test_ignore_les_tirages_rates(self):
        # Deux hallucinations parmi dix mesures ne doivent pas tirer le débit vers le bas.
        mesures = [("x" * 90, 5.0)] * 8 + [("x" * 90, 20.0)] * 2
        assert quality.calibrate(mesures) == 18.0

    def test_refuse_un_echantillon_trop_maigre(self):
        assert quality.calibrate([("x" * 90, 5.0)] * 3) is None


class TestVoixDuMoteur:
    """Une voix appartient à un moteur : la confusion doit se dire, pas se subir."""

    def test_voix_etrangere_refusee_avant_tout_chargement(self):
        from voxlibris.tts.backends import UnknownVoice, load

        with pytest.raises(UnknownVoice) as erreur:
            load("kokoro", "Damien Black", "cpu")
        assert "ff_siwis" in str(erreur.value)

    def test_voix_connue_acceptee(self):
        from voxlibris.tts.backends import BACKENDS

        assert "fr_FR-tom-medium" in BACKENDS["piper"].catalogue

    def test_identifiant_approximatif_resolu(self):
        """« FR-tom-medium » ou « tom » désignent la même voix, sans ambiguïté."""
        from voxlibris.tts.backends import PIPER_VOICES, resolve_voice

        for saisie in ("fr_FR-tom-medium", "FR-tom-medium", "tom", "Fr_FR-Tom-Medium"):
            assert resolve_voice(saisie, PIPER_VOICES) == "fr_FR-tom-medium"

    def test_saisie_ambigue_refusee(self):
        """« medium » convient aux trois voix : c'est à l'utilisateur de trancher."""
        from voxlibris.tts.backends import PIPER_VOICES, resolve_voice

        assert resolve_voice("medium", PIPER_VOICES) is None
        assert resolve_voice("inexistante", PIPER_VOICES) is None


class TestEchantillonDeCalibration:
    def test_reparti_sur_tout_le_chapitre(self):
        """Les douze premiers segments peuvent être une notice en anglais : on étale."""
        from voxlibris.tts.synth import spread

        picked = spread(list(range(100)), 12)
        assert len(picked) == 12
        assert picked[0] == 0 and picked[-1] >= 90
        assert all(b - a >= 7 for a, b in zip(picked, picked[1:], strict=False))

    def test_liste_courte_rendue_entiere(self):
        from voxlibris.tts.synth import spread

        assert spread([1, 2, 3], 12) == [1, 2, 3]


class TestPiste:
    def test_la_piste_ouvre_sur_un_silence(self):
        """Chaque piste commence par un temps de rien, et le manifeste en tient compte."""
        from types import SimpleNamespace

        from voxlibris.tts import synth

        segment = synth.Segment(0, "Le gardien du phare notait la couleur du ciel.", 380, 1, "")
        moteur = SimpleNamespace(say=lambda _: speech(len(segment.text) / 18))
        result = synth.synthesize_chapter(moteur, [segment], PROFILE)  # type: ignore[arg-type]
        lead = synth.LEAD_IN_MS / 1000
        assert result.timing[0]["start"] == pytest.approx(lead, abs=0.01)
        assert result.duration == pytest.approx(lead + len(segment.text) / 18 + 0.38, abs=0.05)
        assert not np.any(result.audio[: int(quality.SAMPLE_RATE * lead) - 1])
