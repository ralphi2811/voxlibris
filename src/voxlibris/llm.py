"""Accès à un modèle de langage, par le protocole OpenAI.

Un seul dialecte est parlé ici : celui de `/v1/chat/completions`. C'est un choix
délibéré, car il est aujourd'hui commun à Ollama, llama.cpp, vLLM, LM Studio,
OpenRouter et OpenAI lui-même. Changer de fournisseur ne demande donc qu'une adresse
et un nom de modèle — deux lignes dans un `.env` — sans aucune ligne de code.

    VOXLIBRIS_LLM_BASE_URL=http://localhost:11434/v1   # Ollama
    VOXLIBRIS_LLM_MODEL=gemma4:12b

Rien n'est installé pour cela : la requête est un POST JSON, servi par la
bibliothèque standard. Le paquet reste ainsi léger, ce qui compte pour l'image web.

L'adresse par défaut est locale, conformément à la promesse du projet : aucun texte ne
quitte la machine tant que l'utilisateur n'a pas lui-même désigné un service distant.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "gemma4:12b"

# Le modèle doit rendre un verdict, pas rédiger : température nulle, réponses courtes.
# La reproductibilité importe ici plus qu'ailleurs, puisque c'est ce qui rend le banc
# d'essai comparable d'une exécution à l'autre.
DEFAULT_TEMPERATURE = 0.0

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
THINK = re.compile(r"<think>.*?</think>", re.S)

# Beaucoup de modèles récents raisonnent avant de répondre. C'est ici du gaspillage, et
# pire : gemma4:12b consacre plusieurs minutes à peser chaque mot, épuise son budget de
# jetons et renvoie un contenu vide. La tâche est un arbitrage bref sur un mot, pas un
# problème à démonter. Le champ est celui d'OpenAI, qu'Ollama reconnaît également ; les
# services qui l'ignorent le rejettent poliment, et la requête est alors rejouée sans lui.
DEFAULT_REASONING = "none"


class LLMError(RuntimeError):
    """Le modèle est injoignable, ou sa réponse est inexploitable."""


def _flag(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "non", "off"}


@dataclass(frozen=True)
class LLMConfig:
    """Configuration du modèle, lue dans l'environnement."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    timeout: float = 180.0
    temperature: float = DEFAULT_TEMPERATURE
    batch_size: int = 12
    enabled: bool = True
    reasoning: str = DEFAULT_REASONING

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LLMConfig:
        if env is None:
            _load_dotenv()
            env = dict(os.environ)
        return cls(
            base_url=env.get("VOXLIBRIS_LLM_BASE_URL") or DEFAULT_BASE_URL,
            model=env.get("VOXLIBRIS_LLM_MODEL") or DEFAULT_MODEL,
            api_key=env.get("VOXLIBRIS_LLM_API_KEY", ""),
            timeout=float(env.get("VOXLIBRIS_LLM_TIMEOUT") or 180),
            temperature=float(env.get("VOXLIBRIS_LLM_TEMPERATURE") or DEFAULT_TEMPERATURE),
            batch_size=int(env.get("VOXLIBRIS_LLM_BATCH") or 12),
            enabled=_flag(env.get("VOXLIBRIS_LLM_ENABLED"), True),
            reasoning=env.get("VOXLIBRIS_LLM_REASONING", DEFAULT_REASONING),
        )

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


def _load_dotenv() -> None:
    """Charge un `.env` s'il existe, sans écraser l'environnement déjà en place.

    Les variables réellement exportées priment donc sur le fichier, ce qui permet à
    Docker Compose et au shell de surcharger la configuration du dépôt.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dépendance optionnelle en développement
        return
    load_dotenv(override=False)


class LLM:
    """Client minimal, sans état, vers un service compatible OpenAI."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._reasoning = bool(self.config.reasoning)

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {self.config.api_key}"}
                    if self.config.api_key
                    else {}
                ),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:400]
            raise LLMError(f"{self.config.endpoint} a répondu {error.code} : {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LLMError(f"{self.config.endpoint} injoignable : {error}") from error

    def complete(self, prompt: str, system: str = "") -> str:
        """Une requête, une réponse. Le contenu du message d'assistant."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": False,
        }
        if self._reasoning:
            payload["reasoning_effort"] = self.config.reasoning

        try:
            body = self._post(payload)
        except LLMError:
            if not self._reasoning:
                raise
            # Certains services refusent ce champ. On l'abandonne pour de bon plutôt que
            # de payer un aller-retour perdu à chaque requête.
            self._reasoning = False
            payload.pop("reasoning_effort")
            body = self._post(payload)

        try:
            content = body["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError) as error:
            raise LLMError(f"Réponse inattendue : {json.dumps(body)[:400]}") from error
        # Les modèles qui raisonnent en clair enferment leur monologue dans des balises,
        # au milieu du contenu utile.
        return THINK.sub("", content).strip()

    def ask_json(self, prompt: str, system: str = "") -> object:
        """Comme `complete`, mais la réponse est analysée comme du JSON.

        Les modèles encadrent volontiers leur JSON de balises Markdown, ou le font
        précéder d'une phrase d'introduction. Plutôt que d'exiger d'eux une discipline
        qu'ils n'ont pas, on extrait la première structure bien formée.
        """
        raw = self.complete(prompt, system)
        text = raw.strip()
        fenced = FENCE.search(text)
        if fenced:
            text = fenced.group(1).strip()
        else:
            start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=-1)
            end = max(text.rfind("]"), text.rfind("}"))
            if start >= 0 and end > start:
                text = text[start : end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise LLMError(f"JSON illisible dans : {raw[:400]}") from error

    def probe(self) -> str:
        """Vérifie que le service répond, et renvoie une phrase à afficher."""
        answer = self.complete("Réponds exactement : prêt", system="Tu réponds en un mot.")
        return f"{self.config.model} sur {self.config.base_url} → « {answer.strip()[:40]} »"
