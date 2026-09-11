"""ZONOS2, éprouvé sans serveur.

Ce qui se teste ici est ce qui casse en pratique : le décodage du flux PCM brut, la
forme des requêtes — l'entrée native plutôt que le protocole d'OpenAI, la langue de
normalisation, le débit envoyé seulement quand il change — et le refus de démarrer sans
adresse, qui évite d'aller chercher un serveur qui n'existe pas.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxlibris.tts import zonos2


def make_pcm(seconds: float = 0.5, rate: int = 44100) -> bytes:
    samples = np.zeros(int(rate * seconds), dtype="<f4")
    samples[::2] = 0.25
    return samples.tobytes()


class FakeClient(zonos2.Client):
    """Le client, avec la couche HTTP remplacée par des réponses écrites d'avance."""

    def __init__(self, responses: dict[str, tuple[bytes, dict]] | None = None):
        self.url, self.timeout = "http://zonos2:1919", 1.0
        self.responses = responses or {}
        self.calls: list[tuple[str, dict | None]] = []

    def _request(self, path, payload=None):
        self.calls.append((path, payload))
        return self.responses[path]


SPEAKERS = (
    b'{"speakers": [{"id": "default_abc", "label": "Marie", "is_default": true},'
    b' {"id": "default_def", "label": "American Female"}]}'
)
CAPABILITIES = b'{"speaker_enabled": true, "speaking_rate_enabled": true}'


class TestDecodage:
    def test_flottants_mono(self):
        audio, rate = zonos2.decode_pcm(make_pcm(0.5, 44100), 44100)
        assert rate == 44100 and len(audio) == 22050
        assert audio.dtype == np.float32 and audio.max() == pytest.approx(0.25)

    def test_flux_tronque_signale(self):
        with pytest.raises(zonos2.Zonos2Error, match="tronqué"):
            zonos2.decode_pcm(b"\x00\x00\x00", 44100)


class TestLangue:
    def test_codes_du_serveur(self):
        assert zonos2.language_code("fr") == "fr_fr"
        assert zonos2.language_code("fr-FR") == "fr_fr"
        assert zonos2.language_code("EN") == "en_us"
        assert zonos2.language_code("pt") == "pt_br"
        assert zonos2.language_code("zh") == "cmn"

    def test_langue_inconnue(self):
        assert zonos2.language_code("eo") is None
        assert zonos2.language_code("") is None


class TestClient:
    def test_sans_adresse_rien_ne_part(self, monkeypatch):
        monkeypatch.setenv("VOXLIBRIS_ZONOS2_BASE_URL", "")
        with pytest.raises(zonos2.Zonos2Error, match="VOXLIBRIS_ZONOS2_BASE_URL"):
            zonos2.Client()

    def test_adresse_sans_barre_finale(self, monkeypatch):
        monkeypatch.setenv("VOXLIBRIS_ZONOS2_BASE_URL", "http://zonos2:1919/")
        assert zonos2.Client().url == "http://zonos2:1919"

    def test_liste_des_voix(self):
        client = FakeClient({"/tts/speakers": (SPEAKERS, {})})
        assert client.speakers() == [
            {"id": "default_abc", "label": "Marie"},
            {"id": "default_def", "label": "American Female"},
        ]
        assert zonos2.voice_names(client) == ["Marie", "American Female"]

    def test_la_sonde_lit_les_capacites(self):
        client = FakeClient({"/tts/capabilities": (CAPABILITIES, {})})
        assert client.probe()["speaker_enabled"] is True

    def test_la_parole_passe_par_l_entree_native(self):
        pcm = make_pcm(0.2, 44100)
        client = FakeClient({"/tts/generate": (pcm, {"x-audio-sample-rate": "44100"})})
        audio, rate = client.speak("Bonjour.", "default_abc", language="fr")
        assert rate == 44100 and len(audio) == 8820
        path, payload = client.calls[-1]
        assert path == "/tts/generate"
        assert payload["speaker_embedding_id"] == "default_abc"
        assert payload["language"] == "fr_fr"
        assert payload["accurate_mode"] is True and payload["stream"] is False
        # Le mode qualité : un enregistrement de studio demandé au modèle.
        assert payload["quality_values"] == zonos2.QUALITY and payload["quality_buckets"] is None
        assert payload["clean_speaker_background"] is True
        # Débit inchangé : le champ n'est pas envoyé, le serveur garde le sien.
        assert "speed" not in payload and "speaking_rate_enabled" not in payload

    def test_le_debit_est_envoye_quand_il_change(self):
        client = FakeClient({"/tts/generate": (make_pcm(0.1), {})})
        client.speak("Bonjour.", "default_abc", speed=0.9)
        payload = client.calls[-1][1]
        assert payload["speed"] == 0.9 and payload["speaking_rate_enabled"] is True

    def test_langue_inconnue_sans_normalisation(self):
        client = FakeClient({"/tts/generate": (make_pcm(0.1), {})})
        client.speak("Saluton.", "default_abc", language="eo")
        payload = client.calls[-1][1]
        assert "language" not in payload and payload["text_normalization"] is False

    def test_frequence_lue_dans_l_en_tete(self):
        client = FakeClient(
            {"/tts/generate": (make_pcm(0.1, 24000), {"x-audio-sample-rate": "24000"})}
        )
        _, rate = client.speak("Bonjour.", "default_abc")
        assert rate == 24000

    def test_reponse_vide(self):
        client = FakeClient({"/tts/generate": (b"", {})})
        with pytest.raises(zonos2.Zonos2Error, match="sans audio"):
            client.speak("Bonjour.", "default_abc")


class TestMoteur:
    def _fake(self, monkeypatch, capabilities=CAPABILITIES):
        client = FakeClient(
            {
                "/tts/capabilities": (capabilities, {}),
                "/tts/speakers": (SPEAKERS, {}),
                "/tts/generate": (make_pcm(0.3, 44100), {"x-audio-sample-rate": "44100"}),
            }
        )
        monkeypatch.setattr("voxlibris.tts.zonos2.Client", lambda *a, **k: client)
        return client

    def test_retrouve_la_voix_par_son_intitule(self, monkeypatch):
        from voxlibris.tts import backends

        client = self._fake(monkeypatch)
        engine = backends.Zonos2Backend(voice="marie", speed=0.9)
        assert engine.voice == "Marie" and engine.supports_speed and engine.speed == 0.9
        assert engine.voices() == ["American Female", "Marie"]
        audio = engine.say("Bonjour.")
        # Rééchantillonné à la fréquence commune des moteurs.
        assert len(audio) == pytest.approx(0.3 * backends.SAMPLE_RATE, abs=2)
        assert client.calls[-1][1]["speaker_embedding_id"] == "default_abc"

    def test_premiere_voix_par_defaut(self, monkeypatch):
        from voxlibris.tts import backends

        self._fake(monkeypatch)
        assert backends.Zonos2Backend().voice == "Marie"

    def test_voix_inconnue_nommee(self, monkeypatch):
        from voxlibris.tts import backends

        self._fake(monkeypatch)
        with pytest.raises(backends.UnknownVoice, match="Marie"):
            backends.Zonos2Backend(voice="Gérard")

    def test_sans_voix_le_remede_est_dit(self, monkeypatch):
        from voxlibris.tts import backends

        client = self._fake(monkeypatch)
        client.responses["/tts/speakers"] = (b'{"speakers": []}', {})
        with pytest.raises(zonos2.Zonos2Error, match="page Voix"):
            backends.Zonos2Backend()

    def test_un_modele_sans_debit_ne_le_promet_pas(self, monkeypatch):
        from voxlibris.tts import backends

        self._fake(monkeypatch, b'{"speaker_enabled": true, "speaking_rate_enabled": false}')
        engine = backends.Zonos2Backend(speed=0.8)
        assert not engine.supports_speed and engine.speed == 1.0

    def test_le_moteur_est_enregistre(self):
        from voxlibris.tts.backends import BACKENDS

        assert BACKENDS["zonos2"].supports_speed is True


class TestAtelier:
    def test_disponible_des_qu_une_adresse_est_donnee(self, monkeypatch):
        from voxlibris import atelier

        monkeypatch.setenv("VOXLIBRIS_ZONOS2_BASE_URL", "")
        assert atelier.engines()["zonos2"] is False
        monkeypatch.setenv("VOXLIBRIS_ZONOS2_BASE_URL", "http://zonos2:1919")
        assert atelier.engines()["zonos2"] is True

    def test_le_battement_suit_le_reglage_sans_relance(self, tmp_path, monkeypatch):
        import json

        from voxlibris import atelier

        monkeypatch.setattr(atelier, "hardware", lambda: {"device": "cpu", "gpu": ""})
        monkeypatch.setenv("VOXLIBRIS_ZONOS2_BASE_URL", "")
        static = {"engines": atelier.installed()}
        atelier.beat(tmp_path, None, static)
        seen = json.loads(atelier.heartbeat_file(tmp_path).read_text())
        assert seen["engines"]["zonos2"] is False and "piper" in seen["engines"]

        monkeypatch.setenv("VOXLIBRIS_ZONOS2_BASE_URL", "http://zonos2:1919")
        atelier.beat(tmp_path, None, static)
        seen = json.loads(atelier.heartbeat_file(tmp_path).read_text())
        assert seen["engines"]["zonos2"] is True


class TestInjoignable:
    def test_localhost_depuis_un_conteneur(self, monkeypatch):
        monkeypatch.setattr(zonos2, "in_container", lambda: True)
        assert "host.docker.internal:1919" in zonos2.unreachable_hint("http://localhost:1919")
        assert zonos2.unreachable_hint("http://zonos2:1919") == ""

    def test_rien_a_dire_hors_conteneur(self, monkeypatch):
        monkeypatch.setattr(zonos2, "in_container", lambda: False)
        assert zonos2.unreachable_hint("http://localhost:1919") == ""
