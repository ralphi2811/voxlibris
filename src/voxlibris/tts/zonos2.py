"""ZONOS2, de Zyphra, servi chez vous.

Le seul moteur de voxlibris qui clone une voix. Il n'y a rien à entraîner : un extrait
audio de quelques secondes, déposé dans le dossier des voix, et le serveur en tire un
plongement de locuteur. Pas de transcription à fournir, pas de longueur imposée. Le
serveur relit ce dossier à chaque demande de liste : une voix déposée existe aussitôt.

Les poids sont sous Apache 2.0 et le serveur — le dépôt de Zyphra, à une révision fixée
dans l'image Docker — sous MIT : aucune restriction d'usage, contrairement à XTTS et aux
poids de Voxtral. Le clonage, lui, engage celui qui dépose l'extrait : il faut avoir le
droit de se servir de cette voix. Voir NOTICE.md.

Le serveur parle deux dialectes. Le protocole d'OpenAI, mais amputé : il ignore le champ
de la voix et ne prend pas la langue. On lui préfère son entrée native, `/tts/generate`,
qui accepte le locuteur, la langue de normalisation du texte — les nombres et les dates
s'y lisent dans la bonne langue — et le débit. Le son revient en PCM flottant brut, à la
fréquence dite dans un en-tête de réponse.

Le français est en deuxième rang chez Zyphra, derrière l'anglais, le mandarin et le
japonais ; le contrôle qualité par segment rattrape les hallucinations que le rapport
technique reconnaît.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from ..config import setting

DEFAULT_PORT = 1919
DEFAULT_SAMPLE_RATE = 44100

# Langues de normalisation du texte que le serveur connaît, par code de projet. Hors de
# cette liste, le texte part tel quel : la lecture reste possible, les nombres en chiffres
# ne sont plus verbalisés — ce que notre préparation fait déjà pour le français.
LANGUAGES = {
    "fr": "fr_fr",
    "en": "en_us",
    "de": "de",
    "es": "es",
    "it": "it",
    "pt": "pt_br",
    "ja": "ja",
    "zh": "cmn",
    "ko": "ko",
}

# Le modèle est conditionné par la qualité d'enregistrement qu'on lui demande d'imiter :
# niveau, rapport signal sur bruit, bande passante, silences et pauses. Sans consigne, il
# tire ces traits au hasard de ses données, avec le souffle et les hésitations qui vont
# avec. On lui demande un studio : c'est le « mode qualité » du rapport technique de
# Zyphra, qui y fait tomber le taux d'erreur de mots en français de 13 % à 4 %, contre un
# léger recul de la ressemblance à l'extrait. Pour un livre entier, l'échange est bon.
QUALITY = {
    "lufs": -20.0,
    "estimated_snr": 50.0,
    "max_pause": 0.3,
    "estimated_bandlimit_hz": 22000.0,
    "leading_silence_s": 0.05,
    "trailing_silence_s": 0.3,
}

# Ce que le serveur accepte comme extrait de voix ; il décode par ffmpeg.
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".webm")


class Zonos2Error(RuntimeError):
    """Le serveur est injoignable, refuse la requête, ou répond ce qu'on n'attendait pas."""


def base_url(env: dict[str, str] | None = None) -> str:
    raw = (
        env.get("VOXLIBRIS_ZONOS2_BASE_URL", "")
        if env is not None
        else setting("VOXLIBRIS_ZONOS2_BASE_URL")
    )
    return raw.strip().rstrip("/")


def language_code(language: str) -> str | None:
    """« fr », « fr-FR » ou « fra » → « fr_fr » ; inconnu → None."""
    return LANGUAGES.get(language.strip().lower()[:2])


def decode_pcm(raw: bytes, rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Flottants 32 bits petit-boutiste, mono : le seul format que rend le serveur."""
    if len(raw) % 4:
        raise Zonos2Error(f"Flux audio tronqué : {len(raw)} octets, multiple de 4 attendu.")
    return np.frombuffer(raw, dtype="<f4").astype(np.float32), rate


def in_container() -> bool:
    return Path("/.dockerenv").exists()


def unreachable_hint(url: str) -> str:
    """Le piège de « localhost » vu depuis un conteneur, dit en une phrase."""
    host = urllib.parse.urlsplit(url).hostname or ""
    if in_container() and host in {"localhost", "127.0.0.1", "::1"}:
        return (
            " — dans un conteneur, « localhost » désigne le conteneur lui-même. Pour un "
            "serveur qui tourne sur la machine, VOXLIBRIS_ZONOS2_BASE_URL="
            f"http://host.docker.internal:{DEFAULT_PORT} ; pour le service Compose, "
            f"http://zonos2:{DEFAULT_PORT} avec « --profile zonos2 »."
        )
    return ""


class Client:
    """Appels HTTP vers le serveur ZONOS2, sans dépendance ajoutée."""

    def __init__(self, url: str = "", timeout: float = 300.0) -> None:
        self.url = (url or base_url()).rstrip("/")
        self.timeout = timeout
        if not self.url:
            raise Zonos2Error(
                "Aucun serveur ZONOS2. Renseignez VOXLIBRIS_ZONOS2_BASE_URL — "
                f"http://zonos2:{DEFAULT_PORT} pour le service Compose sous "
                "« --profile zonos2 »."
            )

    def _request(self, path: str, payload: dict | None = None) -> tuple[bytes, dict[str, str]]:
        headers = {"Accept": "*/*"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}", data=data, headers=headers, method="POST" if data else "GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read(), {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            raise Zonos2Error(f"{path} a répondu {error.code} : {body[:300]}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise Zonos2Error(
                f"{self.url}{path} injoignable : {error}{unreachable_hint(self.url)}"
            ) from error

    def _json(self, path: str, payload: dict | None = None) -> dict:
        body, _ = self._request(path, payload)
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise Zonos2Error(f"{path} : réponse illisible {body[:200]!r}") from error

    def capabilities(self) -> dict:
        """Ce que le modèle chargé sait faire : locuteur, débit, langues normalisées."""
        return self._json("/tts/capabilities")

    def probe(self) -> dict:
        """Vérifie que le serveur répond, avant de lui confier quoi que ce soit."""
        return self.capabilities()

    def speakers(self) -> list[dict]:
        """Les voix du dossier, telles que le serveur les voit : identifiant et intitulé.

        L'identifiant est une empreinte du chemin du fichier, illisible ; l'intitulé est
        son nom sans extension, tirets et soulignés changés en espaces. C'est lui qu'on
        montre et qu'on retient, l'identifiant ne servant qu'à la requête.
        """
        found = self._json("/tts/speakers").get("speakers", [])
        return [
            {"id": str(s["id"]), "label": str(s.get("label") or s["id"])}
            for s in found
            if isinstance(s, dict) and s.get("id")
        ]

    def speak(
        self, text: str, speaker_id: str, language: str = "fr", speed: float = 1.0
    ) -> tuple[np.ndarray, int]:
        payload: dict = {
            "text": text,
            "speaker_embedding_id": speaker_id,
            # Le mode « fidèle » colle au timbre de l'extrait ; l'autre, plus expressif,
            # s'en éloigne. Un livre se lit d'une voix reconnaissable d'un bout à l'autre.
            "accurate_mode": True,
            "quality_values": dict(QUALITY),
            # Le serveur a une consigne par défaut sous l'autre forme, et refuse les deux
            # à la fois : on l'efface explicitement.
            "quality_buckets": None,
            # L'extrait déposé est tenu pour propre : c'est ce que la page Voix demande.
            "clean_speaker_background": True,
            "stream": False,
        }
        code = language_code(language)
        if code:
            payload["language"] = code
        else:
            payload["text_normalization"] = False
        if speed != 1.0:
            payload["speaking_rate_enabled"] = True
            payload["speed"] = float(speed)
        body, headers = self._request("/tts/generate", payload)
        if not body:
            raise Zonos2Error("Réponse sans audio.")
        try:
            rate = int(headers.get("x-audio-sample-rate", DEFAULT_SAMPLE_RATE))
        except ValueError:
            rate = DEFAULT_SAMPLE_RATE
        return decode_pcm(body, rate)


def voice_names(client: Client | None = None) -> list[str]:
    """Intitulés des voix du serveur, dans l'ordre du dossier."""
    return [s["label"] for s in (client or Client()).speakers()]
