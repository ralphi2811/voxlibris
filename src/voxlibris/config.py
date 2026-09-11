"""Configuration de voxlibris : le `.env`, et par-dessus, les réglages faits dans l'interface.

Trois couches, de la plus faible à la plus forte :

1. le fichier `.env` du dépôt, chargé une fois sans écraser l'environnement ;
2. les variables réellement exportées — c'est ainsi que Docker Compose et le shell
   surchargent la configuration ;
3. `settings.json` dans le dossier des données, écrit par la page Réglages.

La troisième couche existe pour une raison précise : un conteneur lit son environnement
au démarrage, et une clé changée dans le `.env` n'y entre qu'après un redémarrage. Le
fichier de réglages, lui, vit dans le volume partagé par l'interface et l'atelier, et
chacun le relit à chaque accès — ce qui coûte un `stat`, rien de plus.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_loaded = False
_settings_cache: tuple[float, dict[str, str]] = (-1.0, {})

# Ce que la page Réglages a le droit d'écrire. Tout le reste reste affaire de `.env`.
EDITABLE = (
    "VOXLIBRIS_LLM_ENABLED",
    "VOXLIBRIS_LLM_BASE_URL",
    "VOXLIBRIS_LLM_MODEL",
    "VOXLIBRIS_LLM_API_KEY",
    "VOXLIBRIS_LLM_BATCH",
    "VOXLIBRIS_LLM_TIMEOUT",
    "VOXLIBRIS_LLM_REASONING",
    "VOXLIBRIS_MISTRAL_API_KEY",
    "VOXLIBRIS_MISTRAL_BASE_URL",
    "VOXLIBRIS_ZONOS2_BASE_URL",
    "VOXLIBRIS_OMNIVOICE_BASE_URL",
    "COQUI_TOS_AGREED",
    "VOXLIBRIS_DEVICE",
)

# Ceux-là ne s'affichent jamais en clair.
SECRETS = ("VOXLIBRIS_LLM_API_KEY", "VOXLIBRIS_MISTRAL_API_KEY")


def data_dir() -> Path:
    """Dossier où vivent les projets, la file et les réglages."""
    return Path(os.environ.get("VOXLIBRIS_DATA", "data")).expanduser().resolve()


def settings_file() -> Path:
    return data_dir() / "settings.json"


def voices_dir() -> Path:
    """Dossier des extraits de voix à cloner ; ZONOS2 et OmniVoice le lisent tel quel."""
    return data_dir() / "voices"


def load_env() -> None:
    """Charge le `.env` une fois pour toutes, sans écraser l'environnement en place."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dépendance optionnelle en développement
        return
    load_dotenv(override=False)


def read_settings() -> dict[str, str]:
    """Les réglages faits dans l'interface, relus dès que le fichier change."""
    global _settings_cache
    path = settings_file()
    try:
        stamp = path.stat().st_mtime
    except FileNotFoundError:
        _settings_cache = (-1.0, {})
        return {}
    if stamp != _settings_cache[0]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raw = {}
        values = {k: str(v) for k, v in raw.items() if k in EDITABLE and v not in (None, "")}
        _settings_cache = (stamp, values)
    return _settings_cache[1]


def write_settings(values: dict[str, str]) -> None:
    """Enregistre les réglages de l'interface. Une valeur vide efface le réglage."""
    current = dict(read_settings())
    for key, value in values.items():
        if key not in EDITABLE:
            continue
        if value:
            current[key] = value
        else:
            current.pop(key, None)
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8")


def setting(name: str, default: str = "") -> str:
    """Valeur d'un réglage : l'interface d'abord, l'environnement et le `.env` ensuite."""
    load_env()
    if (value := read_settings().get(name)) is not None:
        return value
    return os.environ.get(name, default)


def origin(name: str) -> str:
    """D'où vient la valeur en vigueur : « interface », « environnement » ou « — »."""
    load_env()
    if name in read_settings():
        return "interface"
    if os.environ.get(name):
        return "environnement"
    return "—"
