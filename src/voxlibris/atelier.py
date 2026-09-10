"""L'atelier vu de l'extérieur : un battement sur disque, que l'interface relit.

L'interface et l'atelier sont deux processus, souvent deux conteneurs, sans autre lien
que le dossier des données. Pour dire « l'atelier est là, il a une carte, il sait faire
XTTS », il suffit qu'il l'écrive à intervalle régulier dans un petit fichier, et que
l'interface regarde l'âge de ce fichier. Pas de socket, pas de service de plus.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import threading
import time
from pathlib import Path

HEARTBEAT = 5.0
# Au-delà, l'atelier est tenu pour absent : deux battements manqués, avec de la marge.
STALE = 20.0


def heartbeat_file(root: Path) -> Path:
    return root / "atelier.json"


def engines() -> dict[str, bool]:
    """Les moteurs que cet atelier peut charger, sans les charger."""
    from .tts.voxtral import api_key, base_url

    return {
        "xtts": importlib.util.find_spec("TTS") is not None,
        "kokoro": importlib.util.find_spec("kokoro") is not None,
        "piper": importlib.util.find_spec("piper") is not None,
        # Voxtral est joignable soit chez soi, soit avec une clé : dans les deux cas
        # c'est une adresse, pas un paquet, qui le rend disponible.
        "voxtral": bool(api_key()) or "api.mistral.ai" not in base_url(),
    }


def hardware() -> dict[str, object]:
    """La carte graphique, si PyTorch la voit. Sinon, le processeur."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {
                "device": "cuda",
                "gpu": torch.cuda.get_device_name(0),
                "vram_free_gb": round(free / 1e9, 1),
                "vram_total_gb": round(total / 1e9, 1),
            }
    except Exception:
        pass
    return {"device": "cpu", "gpu": "", "vram_free_gb": None, "vram_total_gb": None}


def beat(root: Path, busy: int | None = None, static: dict | None = None) -> None:
    """Écrit le battement. Le renommage rend l'écriture atomique pour le lecteur."""
    payload = {
        "time": time.time(),
        "pid": os.getpid(),
        "busy": busy,
        **(static or {"engines": engines()}),
        **hardware(),
    }
    target = heartbeat_file(root)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, target)


def status(root: Path) -> dict[str, object]:
    """Ce que l'interface affiche : présent ou non, et ce qu'il sait faire."""
    absent = {"online": False, "age": None, "engines": {}, "device": "", "gpu": "", "busy": None}
    path = heartbeat_file(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return absent
    age = time.time() - float(payload.get("time", 0))
    return {**absent, **payload, "age": round(age), "online": age < STALE}


class Pulse(threading.Thread):
    """Bat en arrière-plan pendant que l'atelier travaille.

    Le travail lui-même est synchrone — une synthèse tient un cœur occupé pendant des
    minutes — et ne peut pas battre de lui-même : un fil à part s'en charge.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(daemon=True, name="atelier-pulse")
        self.root = root
        self.busy: int | None = None
        self._static = {"engines": engines()}
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(OSError):
                beat(self.root, self.busy, self._static)
            self._stop.wait(HEARTBEAT)

    def stop(self) -> None:
        self._stop.set()
