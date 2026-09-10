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
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import numpy as np

from ..config import setting

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MODEL = "voxtral-mini-tts-2603"
# Servi par vLLM, le modèle porte le nom de son dépôt Hugging Face.
LOCAL_MODEL = "mistralai/Voxtral-4B-TTS-2603"

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


class Refused(VoxtralError):
    """Un filtre de modération a refusé le texte.

    C'est la limite propre au moteur distant, et elle n'a rien d'anecdotique : un
    classificateur appliqué à de la littérature se trompe. Sur une biographie de Louis
    Braille destinée aux enfants, il a refusé le passage décrivant les préjugés d'un
    personnage à l'égard des aveugles — préjugés que le livre raconte pour les combattre.

    Un moteur local ne connaît pas cette barrière. Elle est le prix de l'API.
    """

    def __init__(self, categories: list[str], text: str = "") -> None:
        self.categories = categories
        self.text = text
        motifs = ", ".join(categories) or "motif non précisé"
        super().__init__(f"texte refusé par la modération de Mistral ({motifs})")


# Le service de modération, interrogeable seul. C'est ce qui permet de savoir en quelques
# secondes ce qui sera refusé, plutôt que de le découvrir au bout de vingt minutes de
# synthèse — et après avoir payé les segments qui, eux, sont passés.
MODERATION_MODEL = "mistral-moderation-latest"
MODERATION_BATCH = 32

# Voix livrées avec les poids, sous forme de plongements déjà calculés. Elles n'ont rien
# à voir avec le catalogue de l'API — celui-ci décline des personnages en émotions, alors
# que le dépôt range par langue. Deux d'entre elles sont françaises.
LOCAL_VOICES = (
    "fr_female",
    "fr_male",
    "neutral_female",
    "neutral_male",
    "casual_female",
    "casual_male",
    "cheerful_female",
    "de_female",
    "de_male",
    "es_female",
    "es_male",
    "it_female",
    "it_male",
    "nl_female",
    "nl_male",
    "pt_female",
    "pt_male",
    "hi_female",
    "hi_male",
    "ar_male",
)


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


def _violated(body: str) -> list[str]:
    """Extrait les catégories enfreintes d'un refus, quelle que soit sa profondeur."""

    def dig(node) -> list[str]:
        if isinstance(node, dict):
            if "violated" in node:
                return []
            found = []
            for key, value in node.items():
                if isinstance(value, dict) and value.get("violated"):
                    found.append(key)
                else:
                    found += dig(value)
            return found
        if isinstance(node, list):
            return [name for item in node for name in dig(item)]
        return []

    try:
        return sorted(set(dig(json.loads(body))))
    except json.JSONDecodeError:
        return []


def in_container() -> bool:
    """Vrai sous Docker. Le fichier est posé par le moteur, quel que soit l'image."""
    return Path("/.dockerenv").exists()


def unreachable_hint(url: str) -> str:
    """Ce qu'il faut savoir quand « localhost » ne répond pas depuis un conteneur.

    Cas vécu : le .env de l'hôte disait localhost:8600, le serveur y tournait, et
    l'atelier sous Compose s'est vu refuser la connexion — pour lui, localhost, c'est
    lui-même. Une trace de quarante lignes ne le dit pas ; cette phrase, si.
    """
    host = urllib.parse.urlsplit(url).hostname or ""
    if in_container() and host in {"localhost", "127.0.0.1", "::1"}:
        return (
            " — dans un conteneur, « localhost » désigne le conteneur lui-même. Pour un "
            "serveur qui tourne sur la machine, VOXLIBRIS_MISTRAL_BASE_URL="
            "http://host.docker.internal:8600/v1 ; pour le service Compose, "
            "http://voxtral:8600/v1 avec « --profile voxtral »."
        )
    return ""


class Client:
    """Appels HTTP vers l'API audio de Mistral, sans dépendance ajoutée."""

    def __init__(self, key: str = "", url: str = "", timeout: float = 120.0) -> None:
        self.key = key or api_key()
        self.url = url or base_url()
        self.timeout = timeout
        if not self.key and not self.is_local:
            raise VoxtralError(
                "Aucune clé d'API Mistral. Renseignez VOXLIBRIS_MISTRAL_API_KEY dans "
                "votre .env — le texte du livre sera alors envoyé à Mistral. Voir "
                "NOTICE.md. Pour faire tourner le modèle chez vous, pointez plutôt "
                "VOXLIBRIS_MISTRAL_BASE_URL sur votre serveur local."
            )

    @property
    def is_local(self) -> bool:
        """Vrai si le service est auto-hébergé, c'est-à-dire n'importe quoi sauf Mistral.

        Les mêmes poids, servis par vLLM, parlent le même protocole : le code ne change
        pas, seule l'adresse. Un serveur à soi n'a ni clé à vérifier, ni catalogue de
        voix enregistrées, ni filtre de modération — c'est tout l'intérêt.

        Le critère est par exclusion, et il le faut : reconnaître « localhost » ne
        suffisait pas, un conteneur joint son voisin par un nom de service — « voxtral »
        — et le client le prenait alors pour l'API de Mistral, cherchant un catalogue de
        compte qui n'existe pas.
        """
        return "api.mistral.ai" not in self.url

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
            body = error.read().decode("utf-8", "replace")
            if error.code == 403 and "guardrail" in body:
                raise Refused(_violated(body)) from error
            raise VoxtralError(f"{path} a répondu {error.code} : {body[:300]}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise VoxtralError(
                f"{self.url}{path} injoignable : {error}{unreachable_hint(self.url)}"
            ) from error

    def probe(self) -> None:
        """Vérifie que le serveur répond, avant de lui confier quoi que ce soit.

        Une adresse fausse se découvre sinon au premier segment de la calibration, au
        fond d'une pile d'appels. Ici elle se découvre en une ligne, tout de suite.
        """
        self._request("/models")

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
        # Un serveur local ne tient pas de catalogue : les voix sont les plongements
        # livrés avec les poids, et l'on ne peut que les nommer.
        if self.is_local:
            return [
                {"id": name, "name": name, "languages": [name.split("_")[0]]}
                for name in LOCAL_VOICES
            ]

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

    def moderate(self, texts: list[str]) -> list[list[str]]:
        """Catégories enfreintes pour chaque texte, sans rien synthétiser.

        Se paie en jetons, non au caractère audio : vérifier un livre entier coûte une
        fraction de centime, là où le découvrir en synthétisant coûte le prix des
        segments déjà produits — et le temps passé à les produire.
        """
        verdicts: list[list[str]] = []
        for start in range(0, len(texts), MODERATION_BATCH):
            lot = texts[start : start + MODERATION_BATCH]
            payload = {"model": MODERATION_MODEL, "input": lot}
            # Un livre entier fait des dizaines de lots, et le service rend parfois un
            # 503 passager. Y renoncer priverait de tout l'avertissement pour un hoquet.
            for attempt in range(3):
                try:
                    body, _ = self._request("/moderations", payload)
                    break
                except VoxtralError:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            for result in json.loads(body).get("results", []):
                categories = result.get("categories", {})
                verdicts.append(sorted(k for k, violated in categories.items() if violated))
        return verdicts

    def speak(self, text: str, voice_id: str, model: str = "") -> tuple[np.ndarray, int]:
        # Les deux services servent les mêmes poids sans nommer les choses pareil :
        # l'API de Mistral désigne une voix enregistrée par « voice_id », vLLM un
        # plongement livré avec le modèle par « voice ».
        payload = {
            "model": model or (LOCAL_MODEL if self.is_local else DEFAULT_MODEL),
            "input": text,
            "response_format": "wav",
            "voice" if self.is_local else "voice_id": voice_id,
        }
        try:
            body, kind = self._request("/audio/speech", payload)
        except Refused as refus:
            # Le refus ne dit pas sur quoi il porte : on le lui rattache ici, faute de
            # quoi le journal signale un blocage sans montrer la phrase en cause.
            refus.text = text
            raise
        # Selon les versions, l'API rend le son encodé dans une enveloppe JSON ou tel
        # quel. On accepte les deux plutôt que de parier sur l'une.
        if kind == "application/json":
            payload = json.loads(body)
            encoded = payload.get("audio_data") or payload.get("audio")
            if not encoded:
                raise VoxtralError(f"Réponse sans audio : {json.dumps(payload)[:200]}")
            body = base64.b64decode(encoded)
        return decode_wav(body)


def voice_names(language: str = "", client: Client | None = None) -> list[str]:
    """Noms des voix du compte, restreints à une langue quand on la connaît.

    Trente voix dont six françaises : les proposer toutes pour un livre français serait
    noyer le choix. Les voix sans langue déclarée — les voix clonées, notamment — sont
    conservées, faute de savoir ce qu'elles valent et dans quelle langue.
    """
    wanted = language.lower()[:2]
    names = [
        voice["name"]
        for voice in (client or Client()).voices()
        if voice.get("name")
        and (
            not wanted
            or not voice.get("languages")
            or any(str(tag).lower().startswith(wanted) for tag in voice["languages"])
        )
    ]
    # Le catalogue décline chaque personnage en émotions — « Marie - Angry », « Marie -
    # Sad »… Un livre se lit d'une voix posée : la variante neutre passe donc devant, et
    # devient le choix par défaut plutôt qu'un hasard alphabétique.
    return sorted(names, key=lambda name: ("neutral" not in name.lower(), name))
