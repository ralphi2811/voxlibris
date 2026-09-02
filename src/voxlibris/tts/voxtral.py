"""Voxtral TTS, par l'API de Mistral.

Seul moteur distant de voxlibris, et la seule entorse à la promesse du « tout en local » :
le texte du livre part chez un tiers, page après page. C'est un choix qui appartient à
l'utilisateur, pas un défaut par commodité — d'où une clé d'API à renseigner soi-même,
sans laquelle le moteur refuse de démarrer.

Ce qu'il apporte en échange : neuf langues, une prosodie de très bon niveau, et surtout
aucune carte graphique. C'est le seul moyen d'obtenir cette qualité sur une machine qui
n'a pas de GPU, là où XTTS en réclame quatre gigaoctets.

Deux points d'attention, développés dans NOTICE.md :

- **Licence.** Les poids ouverts de Voxtral TTS sont sous CC BY-NC 4.0, donc non
  commerciaux. L'usage commercial passe par l'API, qui est payante — c'est exactement
  l'inverse de la situation de XTTS, dont les poids sont gratuits mais restreints.
- **Coût.** 0,016 $ pour mille caractères. Un roman de quatre-vingt mille caractères
  revient à un peu plus d'un dollar, ce que l'interface annonce avant de lancer.

L'API ne propose aucun réglage de vitesse : `speed` est donc sans effet ici, et le dire
vaut mieux que de laisser croire à un réglage qui n'agirait pas.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
import wave

import numpy as np

from ..config import setting

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MODEL = "voxtral-mini-tts-2603"

# L'API recommande de rester sous trois cents mots par requête. Nos segments plafonnent
# à 250 caractères, soit une quarantaine de mots : la marge est confortable.
MAX_WORDS = 300

# Tarif public, en dollars pour mille caractères. Sert à annoncer la dépense avant de la
# faire, jamais à facturer quoi que ce soit.
PRICE_PER_1K_CHARS = 0.016

# Taille de page demandée pour la liste des voix : l'API en sert dix par défaut, et
# refuse au-delà de cent.
PAGE_SIZE = 100


class VoxtralError(RuntimeError):
    """L'API est injoignable, refuse la requête, ou répond ce qu'on n'attendait pas."""


def api_key(env: dict[str, str] | None = None) -> str:
    if env is not None:
        return env.get("VOXLIBRIS_MISTRAL_API_KEY", "").strip()
    return setting("VOXLIBRIS_MISTRAL_API_KEY").strip()


def base_url(env: dict[str, str] | None = None) -> str:
    raw = (
        env.get("VOXLIBRIS_MISTRAL_BASE_URL", "")
        if env is not None
        else setting("VOXLIBRIS_MISTRAL_BASE_URL")
    )
    return (raw or DEFAULT_BASE_URL).rstrip("/")


def estimate_cost(characters: int) -> float:
    return characters / 1000 * PRICE_PER_1K_CHARS


def decode_wav(raw: bytes) -> tuple[np.ndarray, int]:
    """Transforme un WAV en échantillons flottants, avec la bibliothèque standard.

    On demande du WAV plutôt que du MP3 pour cela même : il se décode sans dépendance,
    et il porte sa fréquence d'échantillonnage, qu'aucune documentation ne garantit.
    """
    with wave.open(io.BytesIO(raw)) as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if width != 2:
        raise VoxtralError(f"Échantillons sur {width * 8} bits, attendus sur 16.")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


class Client:
    """Appels HTTP vers l'API audio de Mistral, sans dépendance ajoutée."""

    def __init__(self, key: str = "", url: str = "", timeout: float = 120.0) -> None:
        self.key = key or api_key()
        self.url = url or base_url()
        self.timeout = timeout
        if not self.key:
            raise VoxtralError(
                "Aucune clé d'API Mistral. Renseignez VOXLIBRIS_MISTRAL_API_KEY dans "
                "votre .env — le texte du livre sera alors envoyé à Mistral. Voir "
                "NOTICE.md."
            )

    def _request(self, path: str, payload: dict | None = None) -> tuple[bytes, str]:
        headers = {"Authorization": f"Bearer {self.key}", "Accept": "application/json"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}", data=data, headers=headers, method="POST" if data else "GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read(), response.headers.get_content_type()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300]
            raise VoxtralError(f"{path} a répondu {error.code} : {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise VoxtralError(f"{self.url}{path} injoignable : {error}") from error

    def voices(self) -> list[dict]:
        """Toutes les voix du compte, pagination comprise.

        Deux pièges, découverts en interrogeant l'API plutôt qu'en lisant sa
        documentation. Le premier : la liste est servie par dix, et s'en tenir à la
        première donnait un catalogue exclusivement anglais — alors qu'il compte six voix
        françaises. Une réponse pleine à ras bord est un signal, pas un catalogue.

        Le second : la réponse annonce `page` et `total_pages`, mais ces paramètres-là
        sont ignorés en entrée. Ce sont `offset` et `limit` qui commandent, et `limit`
        est plafonné à cent.
        """
        found: list[dict] = []
        while True:
            body, _ = self._request(f"/audio/voices?offset={len(found)}&limit={PAGE_SIZE}")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as error:
                raise VoxtralError(f"Liste de voix illisible : {body[:200]!r}") from error
            items = payload.get("items", [])
            found += items
            if not items or len(found) >= int(payload.get("total") or len(found)):
                return found

    def speak(self, text: str, voice_id: str, model: str = DEFAULT_MODEL) -> tuple[np.ndarray, int]:
        body, kind = self._request(
            "/audio/speech",
            {"model": model, "input": text, "voice_id": voice_id, "response_format": "wav"},
        )
        # Selon les versions, l'API rend le son encodé dans une enveloppe JSON ou tel
        # quel. On accepte les deux plutôt que de parier sur l'une.
        if kind == "application/json":
            payload = json.loads(body)
            encoded = payload.get("audio_data") or payload.get("audio")
            if not encoded:
                raise VoxtralError(f"Réponse sans audio : {json.dumps(payload)[:200]}")
            body = base64.b64decode(encoded)
        return decode_wav(body)
