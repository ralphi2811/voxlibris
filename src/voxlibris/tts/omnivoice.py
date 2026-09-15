"""OmniVoice, de k2-fsa, servi chez vous.

Le second moteur de voxlibris qui clone une voix, et le plus léger : 0,8 milliard de
paramètres, un modèle de diffusion sur jetons acoustiques, six cents langues, des poids
sous Apache 2.0. Il tient sur une carte de 6 Go et cohabite avec XTTS là où ZONOS2, à
21 Go, exige la carte pour lui seul. En français, son taux d'erreur de mots est un peu
meilleur que celui de ZONOS2 ; sa sortie est à 24 kHz, la sienne à 44,1.

Le paquet ne livre aucun serveur HTTP : celui qu'interroge ce module est le nôtre,
`docker/omnivoice-server.py`, une centaine de lignes de FastAPI dans l'image du profil.
Il lit le même dossier de voix que ZONOS2 — un extrait déposé sert aux deux — et en
tire pour chaque fichier une « invite de clonage » : les jetons de l'extrait et sa
transcription, que Whisper produit à la première demande. Un extrait long est coupé,
au silence le plus net, à une douzaine de secondes : au-delà, le modèle clone moins
bien et va moins vite.

Le son revient en PCM flottant brut, comme chez ZONOS2, à la fréquence dite dans un
en-tête de réponse. Le clonage engage celui qui dépose l'extrait ; voir NOTICE.md.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from ..concierge import implied_url
from ..config import setting
from .zonos2 import AUDIO_EXTENSIONS, in_container

__all__ = ["AUDIO_EXTENSIONS", "Client", "OmnivoiceError", "base_url", "voice_names"]

DEFAULT_PORT = 1920
DEFAULT_SAMPLE_RATE = 24000


class OmnivoiceError(RuntimeError):
    """Le serveur est injoignable, refuse la requête, ou répond ce qu'on n'attendait pas."""


def base_url(env: dict[str, str] | None = None) -> str:
    """Le réglage, sinon l'adresse du service Compose quand son conteneur existe."""
    if env is not None:
        return env.get("VOXLIBRIS_OMNIVOICE_BASE_URL", "").strip().rstrip("/")
    return setting("VOXLIBRIS_OMNIVOICE_BASE_URL").strip().rstrip("/") or implied_url("omnivoice")


def language_code(language: str) -> str | None:
    """« fr », « fr-FR » ou « fra » → « fr » ; vide → None, le modèle devine."""
    code = language.strip().lower()[:2]
    return code if len(code) == 2 and code.isalpha() else None


def decode_pcm(raw: bytes, rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Flottants 32 bits petit-boutiste, mono : le seul format que rend le serveur."""
    if len(raw) % 4:
        raise OmnivoiceError(f"Flux audio tronqué : {len(raw)} octets, multiple de 4 attendu.")
    return np.frombuffer(raw, dtype="<f4").astype(np.float32), rate


def unreachable_hint(url: str) -> str:
    """Le piège de « localhost » vu depuis un conteneur, dit en une phrase."""
    host = urllib.parse.urlsplit(url).hostname or ""
    if in_container() and host in {"localhost", "127.0.0.1", "::1"}:
        return (
            " — dans un conteneur, « localhost » désigne le conteneur lui-même. Pour un "
            "serveur qui tourne sur la machine, VOXLIBRIS_OMNIVOICE_BASE_URL="
            f"http://host.docker.internal:{DEFAULT_PORT} ; pour le service Compose, "
            f"http://omnivoice:{DEFAULT_PORT} avec « --profile omnivoice »."
        )
    return ""


class Client:
    """Appels HTTP vers le serveur OmniVoice de l'atelier, sans dépendance ajoutée."""

    def __init__(self, url: str = "", timeout: float = 300.0) -> None:
        self.url = (url or base_url()).rstrip("/")
        self.timeout = timeout
        if not self.url:
            raise OmnivoiceError(
                "Aucun serveur OmniVoice. Renseignez VOXLIBRIS_OMNIVOICE_BASE_URL — "
                f"http://omnivoice:{DEFAULT_PORT} pour le service Compose sous "
                "« --profile omnivoice »."
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
            raise OmnivoiceError(f"{path} a répondu {error.code} : {body[:300]}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise OmnivoiceError(
                f"{self.url}{path} injoignable : {error}{unreachable_hint(self.url)}"
            ) from error

    def _json(self, path: str, payload: dict | None = None) -> dict:
        body, _ = self._request(path, payload)
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise OmnivoiceError(f"{path} : réponse illisible {body[:200]!r}") from error

    def probe(self) -> dict:
        """Vérifie que le serveur répond et que son modèle est chargé."""
        health = self._json("/health")
        if not health.get("ready", False):
            raise OmnivoiceError("Le serveur OmniVoice répond, mais son modèle n'est pas chargé.")
        return health

    def speakers(self) -> list[dict]:
        """Les voix du dossier, telles que le serveur les nomme : identifiant et intitulé.

        Même convention que ZONOS2 : l'identifiant est une empreinte du nom du fichier,
        l'intitulé son nom sans extension, tirets et soulignés changés en espaces.
        """
        found = self._json("/voices").get("voices", [])
        return [
            {"id": str(v["id"]), "label": str(v.get("label") or v["id"])}
            for v in found
            if isinstance(v, dict) and v.get("id")
        ]

    def speak(
        self, text: str, speaker_id: str, language: str = "fr", speed: float = 1.0
    ) -> tuple[np.ndarray, int]:
        payload: dict = {"text": text, "voice": speaker_id}
        code = language_code(language)
        if code:
            payload["language"] = code
        if speed != 1.0:
            payload["speed"] = float(speed)
        body, headers = self._request("/speak", payload)
        if not body:
            raise OmnivoiceError("Réponse sans audio.")
        try:
            rate = int(headers.get("x-audio-sample-rate", DEFAULT_SAMPLE_RATE))
        except ValueError:
            rate = DEFAULT_SAMPLE_RATE
        return decode_pcm(body, rate)

    def design(
        self, text: str, instruct: str, language: str = "fr", speed: float = 1.0
    ) -> tuple[np.ndarray, int]:
        """Une voix inventée d'après `instruct`, lisant `text`. Nouvelle à chaque appel :
        pour la garder, on enregistre l'extrait et on le clone ensuite."""
        payload: dict = {"text": text, "instruct": instruct}
        code = language_code(language)
        if code:
            payload["language"] = code
        if speed != 1.0:
            payload["speed"] = float(speed)
        body, headers = self._request("/design", payload)
        if not body:
            raise OmnivoiceError("Réponse sans audio.")
        try:
            rate = int(headers.get("x-audio-sample-rate", DEFAULT_SAMPLE_RATE))
        except ValueError:
            rate = DEFAULT_SAMPLE_RATE
        return decode_pcm(body, rate)


# Ce que le modèle sait d'une voix à inventer : ses étiquettes, et leur nom pour nous.
# Les accents ne valent que pour l'anglais, les dialectes que pour le chinois : ils ne
# sont pas proposés.
DESIGN_TRAITS: dict[str, list[tuple[str, str]]] = {
    "gender": [("male", "homme"), ("female", "femme")],
    "age": [
        ("child", "enfant"),
        ("teenager", "adolescent"),
        ("young adult", "jeune adulte"),
        ("middle-aged", "d'âge mûr"),
        ("elderly", "âgé"),
    ],
    "pitch": [
        ("very low pitch", "très grave"),
        ("low pitch", "grave"),
        ("moderate pitch", "médium"),
        ("high pitch", "aiguë"),
        ("very high pitch", "très aiguë"),
    ],
    "style": [("whisper", "chuchotée")],
}

# Ce que la voix inventée lit pour se faire entendre : une douzaine de secondes, de quoi
# la cloner ensuite. Textes originaux, pour ne devoir rien à personne.
DESIGN_TEXTS = {
    "fr": (
        "Le soir tombait sur le port, et les dernières barques rentraient une à une. "
        "Sur le quai, une femme attendait sans impatience, les mains dans les poches de "
        "son manteau, en regardant la lumière décliner sur l'eau."
    ),
    "en": (
        "Evening was settling over the harbour, and the last boats came in one by one. "
        "On the quay a woman waited without impatience, hands deep in her coat pockets, "
        "watching the light fade on the water."
    ),
    "es": (
        "Caía la tarde sobre el puerto y las últimas barcas regresaban una a una. En el "
        "muelle, una mujer esperaba sin impaciencia, con las manos en los bolsillos del "
        "abrigo, mirando cómo la luz se apagaba sobre el agua."
    ),
    "de": (
        "Der Abend senkte sich über den Hafen, und die letzten Boote kehrten eines nach "
        "dem anderen zurück. Am Kai wartete eine Frau ohne Ungeduld, die Hände in den "
        "Manteltaschen, und sah zu, wie das Licht über dem Wasser verblasste."
    ),
    "it": (
        "La sera scendeva sul porto e le ultime barche rientravano una dopo l'altra. Sul "
        "molo una donna aspettava senza impazienza, le mani nelle tasche del cappotto, "
        "guardando la luce spegnersi sull'acqua."
    ),
}


def design_instruct(choices: dict[str, str]) -> str:
    """La description que le modèle attend, à partir d'un choix par trait ; les traits
    laissés vides ne pèsent pas. Une valeur inconnue est refusée."""
    parts = []
    for trait, options in DESIGN_TRAITS.items():
        chosen = (choices.get(trait) or "").strip().lower()
        if not chosen:
            continue
        if chosen not in {value for value, _ in options}:
            raise ValueError(f"Trait inconnu pour {trait} : {chosen!r}")
        parts.append(chosen)
    return ", ".join(parts)


def design_text(language: str) -> str:
    return DESIGN_TEXTS.get(language_code(language) or "fr", DESIGN_TEXTS["fr"])


def describe_instruct(instruct: str) -> str:
    """La description en clair, pour l'afficher : « femme, âgée, grave »."""
    names = {value: label for options in DESIGN_TRAITS.values() for value, label in options}
    return ", ".join(
        names.get(part.strip(), part.strip()) for part in instruct.split(",") if part.strip()
    )


def voice_names(client: Client | None = None) -> list[str]:
    """Intitulés des voix du serveur, dans l'ordre du dossier."""
    return [v["label"] for v in (client or Client()).speakers()]
