"""OmniVoice, éprouvé sans serveur.

Le serveur est le nôtre (docker/omnivoice-server.py) : ce qui se teste ici est le
contrat entre lui et le client — la forme des requêtes, le décodage du flux PCM, la
langue et le débit envoyés seulement quand ils comptent — et le refus de démarrer sans
adresse.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxlibris.tts import omnivoice


def make_pcm(seconds: float = 0.5, rate: int = 24000) -> bytes:
    samples = np.zeros(int(rate * seconds), dtype="<f4")
    samples[::2] = 0.25
    return samples.tobytes()


class FakeClient(omnivoice.Client):
    """Le client, avec la couche HTTP remplacée par des réponses écrites d'avance."""

    def __init__(self, responses: dict[str, tuple[bytes, dict]] | None = None):
        self.url, self.timeout = "http://omnivoice:1920", 1.0
        self.responses = responses or {}
        self.calls: list[tuple[str, dict | None]] = []

    def _request(self, path, payload=None):
        self.calls.append((path, payload))
        return self.responses[path]


VOICES = (
    b'{"voices": [{"id": "voice_abc", "label": "Marie", "file": "marie.wav", "prepared": true},'
    b' {"id": "voice_def", "label": "American Female", "file": "AmericanFemale.mp3"}]}'
)
HEALTH = b'{"ready": true, "model": "k2-fsa/OmniVoice", "device": "cuda", "sample_rate": 24000}'


class TestDecodage:
    def test_flottants_mono(self):
        audio, rate = omnivoice.decode_pcm(make_pcm(0.5), 24000)
        assert rate == 24000 and len(audio) == 12000
        assert audio.dtype == np.float32 and audio.max() == pytest.approx(0.25)

    def test_flux_tronque_signale(self):
        with pytest.raises(omnivoice.OmnivoiceError, match="tronqué"):
            omnivoice.decode_pcm(b"\x00\x00\x00", 24000)


class TestLangue:
    def test_code_a_deux_lettres(self):
        assert omnivoice.language_code("fr") == "fr"
        assert omnivoice.language_code("fr-FR") == "fr"
        assert omnivoice.language_code("EN") == "en"

    def test_vide_laisse_deviner(self):
        assert omnivoice.language_code("") is None
        assert omnivoice.language_code("1") is None


class TestClient:
    def test_sans_adresse_rien_ne_part(self, monkeypatch):
        monkeypatch.setenv("VOXLIBRIS_OMNIVOICE_BASE_URL", "")
        with pytest.raises(omnivoice.OmnivoiceError, match="VOXLIBRIS_OMNIVOICE_BASE_URL"):
            omnivoice.Client()

    def test_adresse_sans_barre_finale(self, monkeypatch):
        monkeypatch.setenv("VOXLIBRIS_OMNIVOICE_BASE_URL", "http://omnivoice:1920/")
        assert omnivoice.Client().url == "http://omnivoice:1920"

    def test_liste_des_voix(self):
        client = FakeClient({"/voices": (VOICES, {})})
        assert client.speakers() == [
            {"id": "voice_abc", "label": "Marie"},
            {"id": "voice_def", "label": "American Female"},
        ]
        assert omnivoice.voice_names(client) == ["Marie", "American Female"]

    def test_la_sonde_exige_un_modele_charge(self):
        client = FakeClient({"/health": (HEALTH, {})})
        assert client.probe()["device"] == "cuda"
        client.responses["/health"] = (b'{"ready": false}', {})
        with pytest.raises(omnivoice.OmnivoiceError, match="pas chargé"):
            client.probe()

    def test_la_parole(self):
        client = FakeClient({"/speak": (make_pcm(0.2), {"x-audio-sample-rate": "24000"})})
        audio, rate = client.speak("Bonjour.", "voice_abc", language="fr")
        assert rate == 24000 and len(audio) == 4800
        path, payload = client.calls[-1]
        assert path == "/speak"
        assert payload == {"text": "Bonjour.", "voice": "voice_abc", "language": "fr"}

    def test_le_debit_est_envoye_quand_il_change(self):
        client = FakeClient({"/speak": (make_pcm(0.1), {})})
        client.speak("Bonjour.", "voice_abc", speed=0.9)
        assert client.calls[-1][1]["speed"] == 0.9

    def test_langue_inconnue_laisse_deviner(self):
        client = FakeClient({"/speak": (make_pcm(0.1), {})})
        client.speak("Saluton.", "voice_abc", language="")
        assert "language" not in client.calls[-1][1]

    def test_reponse_vide(self):
        client = FakeClient({"/speak": (b"", {})})
        with pytest.raises(omnivoice.OmnivoiceError, match="sans audio"):
            client.speak("Bonjour.", "voice_abc")


class TestMoteur:
    def _fake(self, monkeypatch):
        client = FakeClient(
            {
                "/health": (HEALTH, {}),
                "/voices": (VOICES, {}),
                "/speak": (make_pcm(0.3), {"x-audio-sample-rate": "24000"}),
            }
        )
        monkeypatch.setattr("voxlibris.tts.omnivoice.Client", lambda *a, **k: client)
        return client

    def test_retrouve_la_voix_par_son_intitule(self, monkeypatch):
        from voxlibris.tts import backends

        client = self._fake(monkeypatch)
        engine = backends.OmnivoiceBackend(voice="marie", speed=0.9)
        assert engine.voice == "Marie" and engine.supports_speed and engine.speed == 0.9
        assert engine.voices() == ["American Female", "Marie"]
        audio = engine.say("Bonjour.")
        # Rééchantillonné de 24 kHz à la fréquence commune des moteurs.
        assert len(audio) == pytest.approx(0.3 * backends.SAMPLE_RATE, abs=2)
        assert client.calls[-1][1]["voice"] == "voice_abc"

    def test_premiere_voix_par_defaut(self, monkeypatch):
        from voxlibris.tts import backends

        self._fake(monkeypatch)
        assert backends.OmnivoiceBackend().voice == "Marie"

    def test_voix_inconnue_nommee(self, monkeypatch):
        from voxlibris.tts import backends

        self._fake(monkeypatch)
        with pytest.raises(backends.UnknownVoice, match="Marie"):
            backends.OmnivoiceBackend(voice="Gérard")

    def test_sans_voix_le_remede_est_dit(self, monkeypatch):
        from voxlibris.tts import backends

        client = self._fake(monkeypatch)
        client.responses["/voices"] = (b'{"voices": []}', {})
        with pytest.raises(omnivoice.OmnivoiceError, match="page Voix"):
            backends.OmnivoiceBackend()

    def test_le_moteur_est_enregistre(self):
        from voxlibris.tts.backends import BACKENDS

        assert BACKENDS["omnivoice"].supports_speed is True


class TestAtelier:
    def test_disponible_des_qu_une_adresse_est_donnee(self, monkeypatch):
        from voxlibris import atelier

        monkeypatch.setenv("VOXLIBRIS_OMNIVOICE_BASE_URL", "")
        assert atelier.engines()["omnivoice"] is False
        monkeypatch.setenv("VOXLIBRIS_OMNIVOICE_BASE_URL", "http://omnivoice:1920")
        assert atelier.engines()["omnivoice"] is True


class TestInjoignable:
    def test_localhost_depuis_un_conteneur(self, monkeypatch):
        monkeypatch.setattr(omnivoice, "in_container", lambda: True)
        assert "host.docker.internal:1920" in omnivoice.unreachable_hint("http://localhost:1920")
        assert omnivoice.unreachable_hint("http://omnivoice:1920") == ""
