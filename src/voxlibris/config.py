"""Lecture du fichier `.env`, pour tout le projet.

Ce chargement vivait dans le module du modèle de langage, et n'était donc déclenché que
par lui : une clé écrite dans le `.env` restait invisible à la synthèse vocale, qui
annonçait alors qu'aucune clé n'avait été renseignée. Un même fichier de configuration
doit être lu de la même façon quel que soit le module qui en a besoin.

Les variables réellement exportées dans l'environnement priment sur le fichier, ce qui
permet à Docker Compose et au shell de surcharger la configuration du dépôt.
"""

from __future__ import annotations

import os

_loaded = False


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


def setting(name: str, default: str = "") -> str:
    """Valeur d'un réglage, le `.env` étant chargé au premier accès."""
    load_env()
    return os.environ.get(name, default)
