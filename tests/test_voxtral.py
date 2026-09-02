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
        client = FakeClient(
            {"/audio/voices?offset=0&limit=100": (body.encode(), "application/json")}
        )
        assert client.voices() == [{"id": "a1", "name": "Ambre"}]

    def test_toutes_les_pages_sont_lues(self):
        """S'arrêter à la première page masquait les six voix françaises du catalogue."""
        pages = {
            f"/audio/voices?offset={offset}&limit=100": (
                json.dumps(
                    {"items": [{"id": f"v{offset}", "name": f"Voix {offset}"}], "total": 3}
                ).encode(),
                "application/json",
            )
            for offset in (0, 1, 2)
        }
        client = FakeClient(pages)
        assert [v["name"] for v in client.voices()] == ["Voix 0", "Voix 1", "Voix 2"]

    def test_page_vide_arrete_la_lecture(self):
        """Un total incohérent ne doit pas faire tourner la boucle indéfiniment."""
        client = FakeClient(
            {
                "/audio/voices?offset=0&limit=100": (
                    json.dumps({"items": [], "total": 99}).encode(),
                    "application/json",
                )
            }
        )
        assert client.voices() == []

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


class TestModeration:
    def test_categories_extraites_du_refus(self):
        """Le corps d'un 403 imbrique les catégories : il faut aller les chercher."""
        body = json.dumps(
            {
                "message": "Request blocked by guardrail policy",
                "guardrails": [
                    {
                        "moderation_llm_v2": {
                            "action": "block",
                            "categories": {
                                "sexual": {"violated": False},
                                "hate_and_discrimination": {"violated": True},
                            },
                        }
                    }
                ],
            }
        )
        assert voxtral._violated(body) == ["hate_and_discrimination"]

    def test_corps_illisible_ne_casse_rien(self):
        assert voxtral._violated("<html>503</html>") == []

    def test_verdicts_par_lot(self):
        body = json.dumps(
            {
                "results": [
                    {"categories": {"pii": True, "sexual": False}},
                    {"categories": {"pii": False}},
                ]
            }
        )
        client = FakeClient({"/moderations": (body.encode(), "application/json")})
        assert client.moderate(["Je m'appelle Gabriel.", "Il faisait beau."]) == [["pii"], []]


class TestLocal:
    """Les mêmes poids servis par vLLM : même protocole, deux ou trois noms qui diffèrent."""

    def test_adresse_locale_reconnue(self):
        assert voxtral.Client(url="http://localhost:8600/v1").is_local
        assert not voxtral.Client(key="k", url="https://api.mistral.ai/v1").is_local

    def test_pas_de_cle_exigee_en_local(self, monkeypatch):
        """La clé ne protégeait que d'un envoi à un tiers : en local, il n'y en a pas."""
        monkeypatch.setenv("VOXLIBRIS_MISTRAL_API_KEY", "")
        assert voxtral.Client(url="http://127.0.0.1:8600/v1").is_local

    def test_le_champ_de_la_voix_change(self):
        """vLLM attend « voice », l'API de Mistral « voice_id »."""
        client = FakeClient({"/audio/speech": (make_wav(), "audio/wav")})
        client.url = "http://localhost:8600/v1"
        client.speak("Bonjour.", voice_id="fr_female")
        envoi = client.calls[0][1]
        assert envoi["voice"] == "fr_female" and "voice_id" not in envoi
        assert envoi["model"] == voxtral.LOCAL_MODEL

    def test_voix_locales_sans_requete(self):
        """Les plongements sont livrés avec les poids : aucun catalogue à interroger."""
        client = FakeClient({})
        client.url = "http://localhost:8600/v1"
        noms = [v["name"] for v in client.voices()]
        assert "fr_female" in noms and "fr_male" in noms
        assert not client.calls
        assert voxtral.voice_names("fr", client=client) == ["fr_female", "fr_male"]
