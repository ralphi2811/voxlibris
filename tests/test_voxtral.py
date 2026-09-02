"""Le moteur distant, éprouvé sans réseau ni clé.

Ce qui se teste ici est ce qui casse en pratique : le décodage du son, la forme de la
réponse — l'API rend tantôt du JSON, tantôt les octets bruts — et le refus de démarrer
sans clé, qui est la garantie qu'aucun texte ne part par mégarde.
"""

from __future__ import annotations

import base64
import io
import json
import wave

import numpy as np
import pytest

from voxlibris.tts import voxtral


def make_wav(seconds: float = 0.5, rate: int = 24000, channels: int = 1) -> bytes:
    samples = np.zeros(int(rate * seconds) * channels, dtype="<i2")
    samples[:: max(channels, 1)] = 8000
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


class FakeClient(voxtral.Client):
    """Le client, avec la couche HTTP remplacée par des réponses écrites d'avance."""

    def __init__(self, responses: dict[str, tuple[bytes, str]]):
        self.key, self.url, self.timeout = "essai", "http://exemple.invalide/v1", 1.0
        self.responses = responses
        self.calls: list[tuple[str, dict | None]] = []

    def _request(self, path, payload=None):
        self.calls.append((path, payload))
        return self.responses[path]


class TestDecodage:
    def test_wav_mono(self):
        audio, rate = voxtral.decode_wav(make_wav(0.5, 24000))
        assert rate == 24000
        assert len(audio) == pytest.approx(12000, abs=2)
        assert audio.dtype == np.float32
        assert -1.0 <= audio.max() <= 1.0

    def test_stereo_ramene_a_un_canal(self):
        """Les moteurs de voxlibris rendent du mono : deux canaux seraient mal recollés."""
        audio, _ = voxtral.decode_wav(make_wav(0.5, 24000, channels=2))
        assert audio.ndim == 1
        assert len(audio) == pytest.approx(12000, abs=2)

    def test_profondeur_inattendue_signalee(self):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(1)
            handle.setframerate(24000)
            handle.writeframes(b"\x00" * 100)
        with pytest.raises(voxtral.VoxtralError, match="8 bits"):
            voxtral.decode_wav(buffer.getvalue())


class TestClient:
    def test_son_enveloppe_dans_du_json(self):
        payload = json.dumps({"audio_data": base64.b64encode(make_wav()).decode()})
        client = FakeClient({"/audio/speech": (payload.encode(), "application/json")})

        audio, rate = client.speak("Bonjour.", voice_id="v1")
        assert rate == 24000 and len(audio) > 0
        assert client.calls[0][1]["response_format"] == "wav"
        assert client.calls[0][1]["voice_id"] == "v1"

    def test_son_rendu_tel_quel(self):
        """Selon les versions l'API renvoie les octets sans enveloppe : les deux marchent."""
        client = FakeClient({"/audio/speech": (make_wav(), "audio/wav")})
        audio, _ = client.speak("Bonjour.", voice_id="v1")
        assert len(audio) > 0

    def test_reponse_sans_audio(self):
        client = FakeClient({"/audio/speech": (b'{"detail": "quota"}', "application/json")})
        with pytest.raises(voxtral.VoxtralError, match="sans audio"):
            client.speak("Bonjour.", voice_id="v1")

    def test_liste_des_voix(self):
        body = json.dumps({"items": [{"id": "a1", "name": "Ambre"}], "total": 1})
        client = FakeClient({"/audio/voices": (body.encode(), "application/json")})
        assert client.voices() == [{"id": "a1", "name": "Ambre"}]

    def test_sans_cle_rien_ne_part(self, monkeypatch):
        """Le refus de démarrer est ce qui garantit qu'aucun texte ne sort par mégarde."""
        monkeypatch.setenv("VOXLIBRIS_MISTRAL_API_KEY", "")
        with pytest.raises(voxtral.VoxtralError, match="Aucune clé"):
            voxtral.Client()


class TestCout:
    def test_estimation(self):
        assert voxtral.estimate_cost(1000) == pytest.approx(0.016)
        # Un roman de quatre-vingt mille caractères, l'ordre de grandeur annoncé.
        assert voxtral.estimate_cost(80_000) == pytest.approx(1.28)


class TestMoteur:
    def test_declare_ne_pas_savoir_ralentir(self):
        """L'API n'a pas de réglage de débit : mieux vaut le dire que l'ignorer."""
        from voxlibris.tts.backends import BACKENDS

        assert BACKENDS["voxtral"].supports_speed is False
        assert all(BACKENDS[name].supports_speed for name in ("xtts", "kokoro", "piper"))
