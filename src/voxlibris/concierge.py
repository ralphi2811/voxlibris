"""Le concierge de la carte graphique : réveille et endort les serveurs de synthèse.

Trois moteurs vivent dans leur propre conteneur — Voxtral, ZONOS2, OmniVoice — et deux
d'entre eux, une fois chargés, occupent la carte pour eux seuls. Sans concierge, c'est à
vous de choisir un profil au lancement, et de relancer la pile pour en changer. Avec lui,
vous déclarez tous les profils que vous voulez : avant une tâche, l'atelier réveille le
serveur qu'elle demande, endort ceux qui ne tiendraient pas à côté, et rend la carte
après un quart d'heure sans rien faire.

Pour cela, il parle à l'API de Docker — lister les conteneurs du projet Compose, en
démarrer un, en arrêter un. Pas à la socket brute, qui vaut root sur la machine : à un
relais (le service `docker-proxy` du docker-compose.yml) qui n'ouvre que ces trois
verbes. Sans relais, VOXLIBRIS_DOCKER_URL vide, le concierge est inerte : l'atelier voit
ce qui répond et ne touche à rien, comme avant.

Ce qu'il sait des moteurs — leur poids sur la carte, l'adresse de leur service, le temps
d'un réveil — est écrit ici, mesuré sur une RTX 4090. Il en déduit aussi l'adresse d'un
serveur dont le conteneur existe : plus rien à saisir dans les Réglages pour un service
Compose, le champ ne sert qu'à un serveur externe.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from .config import data_dir, setting

log = logging.getLogger("voxlibris.concierge")

Report = Callable[[str], None]

# Étiquettes que Compose pose sur chaque conteneur qu'il crée.
LABEL_PROJECT = "com.docker.compose.project"
LABEL_SERVICE = "com.docker.compose.service"

DEFAULT_IDLE_MINUTES = 15
# Un serveur qui ne répond toujours pas au bout de ce délai a un problème que le
# concierge ne réglera pas en attendant davantage.
WAKE_TIMEOUT = 900.0
# Quand on ne sait pas ce que la carte contient, c'est celle sur laquelle tout a été
# mesuré.
DEFAULT_VRAM_GB = 24.0
# Ce que la carte perd hors modèles : contextes CUDA, affichage, arrondis. Mesuré : ZONOS2
# annoncé à 21 Go ne laisse que 0,6 Go sur une carte de 24,5.
RESERVE_GB = 1.5


@dataclass(frozen=True)
class Served:
    """Un moteur servi par un conteneur du projet Compose."""

    service: str  # nom du service, qui est aussi son nom d'hôte sur le réseau
    origin: str  # adresse du serveur, sans chemin
    url: str  # ce qu'on met dans le réglage du moteur
    probe: str  # chemin qui répond 200 quand le serveur est prêt à parler
    vram_gb: float  # ce qu'il occupe sur la carte une fois chargé
    boot_s: int  # ordre de grandeur d'un réveil, pour prévenir


SERVED: dict[str, Served] = {
    "voxtral": Served(
        "voxtral", "http://voxtral:8600", "http://voxtral:8600/v1", "/v1/models", 16.0, 120
    ),
    "zonos2": Served(
        "zonos2", "http://zonos2:1919", "http://zonos2:1919", "/tts/capabilities", 21.0, 60
    ),
    "omnivoice": Served(
        "omnivoice", "http://omnivoice:1920", "http://omnivoice:1920", "/health", 3.0, 45
    ),
}

# Ce que les moteurs embarqués dans l'atelier prennent sur la carte le temps d'une tâche.
# Ils sont libérés après chaque tâche ; avant un réveil, ils ne comptent donc pas.
EMBEDDED_VRAM_GB = {"xtts": 2.5, "kokoro": 0.5}


class DockerError(RuntimeError):
    """Le relais ne répond pas, ou refuse."""


def docker_url() -> str:
    return setting("VOXLIBRIS_DOCKER_URL").strip().rstrip("/")


def idle_minutes() -> float:
    """Sans rien à faire pendant ce temps, la carte est rendue. Zéro : jamais."""
    raw = setting("VOXLIBRIS_GPU_IDLE_MIN").strip()
    try:
        return max(0.0, float(raw)) if raw else float(DEFAULT_IDLE_MINUTES)
    except ValueError:
        return float(DEFAULT_IDLE_MINUTES)


class Docker:
    """Le strict nécessaire de l'API de Docker, sans dépendance ajoutée."""

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, method: str, path: str, query: dict | None = None) -> tuple[int, bytes]:
        target = f"{self.url}{path}"
        if query:
            target += "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(target, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            # 304 « déjà dans cet état » est une réponse, pas une erreur.
            if error.code == 304:
                return 304, b""
            body = error.read().decode("utf-8", "replace")
            raise DockerError(
                f"Docker a répondu {error.code} à {method} {path} : {body[:200]}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise DockerError(f"relais Docker injoignable ({self.url}) : {error}") from error

    def inspect(self, ident: str) -> dict:
        _, body = self._call("GET", f"/containers/{ident}/json")
        return json.loads(body)

    def containers(self, project: str | None = None) -> list[dict]:
        """Tous les conteneurs, arrêtés compris, d'un projet Compose ou de tous."""
        filters: dict = {"label": [f"{LABEL_PROJECT}={project}" if project else LABEL_SERVICE]}
        _, body = self._call(
            "GET", "/containers/json", {"all": "true", "filters": json.dumps(filters)}
        )
        return json.loads(body)

    def start(self, ident: str) -> None:
        self._call("POST", f"/containers/{ident}/start")

    def stop(self, ident: str, grace_s: int = 30) -> None:
        self._call("POST", f"/containers/{ident}/stop", {"t": str(grace_s)})


def own_project(docker: Docker) -> str | None:
    """Le projet Compose de ce conteneur-ci, lu sur ses propres étiquettes."""
    if name := os.environ.get("COMPOSE_PROJECT_NAME"):
        return name
    try:
        labels = docker.inspect(socket.gethostname()).get("Config", {}).get("Labels", {})
    except (DockerError, ValueError):
        return None
    return labels.get(LABEL_PROJECT) or None


def probe(served: Served, timeout: float = 5.0) -> bool:
    """Vrai quand le serveur répond à sa sonde : chargé, prêt à parler."""
    try:
        with urllib.request.urlopen(served.origin + served.probe, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def read_heartbeat() -> dict:
    """Le dernier battement écrit, tel quel — lu ici sans passer par `atelier`, qui
    importe ce module."""
    try:
        return json.loads((data_dir() / "atelier.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def implied_url(engine: str, served: dict | None = None) -> str:
    """L'adresse du service Compose d'un moteur, si son conteneur existe.

    C'est le battement de l'atelier qui le dit, et l'interface le lit comme l'atelier :
    un conteneur `zonos2` dans le projet, et « http://zonos2:1919 » va de soi. L'atelier
    lui-même passe son inventaire du moment plutôt que de relire son propre battement.
    """
    known = SERVED.get(engine)
    if known is None:
        return ""
    if served is None:
        served = read_heartbeat().get("served") or {}
    return known.url if served.get(engine, {}).get("state") in ("running", "stopped") else ""


def served_state(engine: str) -> str:
    """« running », « stopped », « absent » — ou « » quand rien n'est su."""
    return str((read_heartbeat().get("served") or {}).get(engine, {}).get("state", ""))


class Concierge:
    """Tient l'inventaire des serveurs du projet, et la carte qu'ils se partagent."""

    def __init__(
        self,
        docker: Docker | None,
        project: str | None = None,
        vram_gb: float | None = None,
    ) -> None:
        self.docker = docker
        self.project = project
        self.vram_gb = vram_gb
        self.lock = threading.RLock()
        self.last_activity = time.monotonic()
        self._inventory: tuple[float, dict[str, dict]] = (-1.0, {})
        self.asleep_since: float | None = None

    @classmethod
    def from_env(cls) -> Concierge:
        url = docker_url()
        if not url:
            log.info(
                "pas de relais Docker : les serveurs de synthèse ne seront ni réveillés ni endormis"
            )
            return cls(None)
        docker = Docker(url)
        project = own_project(docker)
        try:
            docker.containers(project)
        except DockerError as error:
            log.warning("relais Docker inutilisable, concierge inerte : %s", error)
            return cls(None)
        log.info("concierge prêt sur %s (projet %s)", url, project or "tous")
        return cls(docker, project)

    @property
    def active(self) -> bool:
        return self.docker is not None

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def idle_for(self) -> float:
        return time.monotonic() - self.last_activity

    # -- inventaire ------------------------------------------------------------------

    def inventory(self, max_age: float = 3.0) -> dict[str, dict]:
        """État de chaque moteur servi : running, stopped ou absent, et son conteneur.

        Relu au plus toutes les trois secondes : le battement le demande souvent, et
        l'API de Docker n'a pas à le payer à chaque fois.
        """
        if not self.docker:
            return {}
        stamp, cached = self._inventory
        if time.monotonic() - stamp < max_age:
            return cached
        found: dict[str, dict] = {engine: {"state": "absent", "name": ""} for engine in SERVED}
        try:
            for container in self.docker.containers(self.project):
                service = (container.get("labels") or container.get("Labels") or {}).get(
                    LABEL_SERVICE, ""
                )
                for engine, served in SERVED.items():
                    if service == served.service:
                        running = container.get("State") == "running"
                        names = container.get("Names") or [""]
                        found[engine] = {
                            "state": "running" if running else "stopped",
                            "name": str(names[0]).lstrip("/"),
                        }
        except DockerError as error:
            log.warning("inventaire impossible : %s", error)
            return cached
        self._inventory = (time.monotonic(), found)
        return found

    def forget(self) -> None:
        self._inventory = (-1.0, {})

    def running(self) -> list[str]:
        return [e for e, s in self.inventory().items() if s["state"] == "running"]

    # -- réveil et sommeil -------------------------------------------------------------

    def total_vram(self) -> float:
        if self.vram_gb:
            return self.vram_gb
        try:
            import torch

            if torch.cuda.is_available():
                self.vram_gb = torch.cuda.mem_get_info()[1] / 1e9
                return self.vram_gb
        except Exception:
            pass
        return DEFAULT_VRAM_GB

    def ensure(
        self, engine: str, report: Report = log.info, free: Callable[[], None] | None = None
    ) -> None:
        """Avant une tâche : le serveur du moteur répond, quitte à en endormir d'autres.

        Un moteur embarqué (XTTS, Kokoro) n'a pas de serveur ; on s'assure seulement que
        les serveurs qui tournent lui laissent la place. Un moteur dont le conteneur
        n'existe pas — serveur externe, ou API distante — est laissé tel quel.
        """
        if not self.docker:
            return
        with self.lock:
            self.touch()
            inventory = self.inventory(max_age=0)
            entry = inventory.get(engine)
            served = SERVED.get(engine)
            if served and entry and entry["state"] == "running":
                return
            if served and (not entry or entry["state"] == "absent"):
                return
            needed = served.vram_gb if served else EMBEDDED_VRAM_GB.get(engine, 0.0)
            self._make_room(engine, needed, inventory, report)
            if not served or not entry:
                return
            if free:
                free()
            report(f"réveil de {engine} — compter {served.boot_s} s")
            self.docker.start(entry["name"])
            self.forget()
            started = time.monotonic()
            last_word = started
            while not probe(served):
                waited = time.monotonic() - started
                if waited > WAKE_TIMEOUT:
                    raise DockerError(
                        f"{engine} ne répond toujours pas après {waited:.0f} s : "
                        f"voir « docker logs {entry['name']} »."
                    )
                if time.monotonic() - last_word >= 15:
                    report(f"réveil de {engine}… {waited:.0f} s")
                    last_word = time.monotonic()
                time.sleep(2.0)
            report(f"{engine} prêt en {time.monotonic() - started:.0f} s")
            self.touch()

    def _make_room(self, engine: str, needed: float, inventory: dict, report: Report) -> None:
        """Endort les serveurs qui gênent, du plus lourd au plus léger, jusqu'à tenir."""
        total = self.total_vram()
        others = [
            e
            for e, s in inventory.items()
            if s["state"] == "running" and e != engine and e in SERVED
        ]
        used = sum(SERVED[e].vram_gb for e in others)
        for other in sorted(others, key=lambda e: -SERVED[e].vram_gb):
            if used + needed + RESERVE_GB <= total:
                break
            report(f"mise en veille de {other} : la carte ne tiendrait pas les deux")
            self.sleep(other)
            used -= SERVED[other].vram_gb

    def sleep(self, engine: str) -> bool:
        """Arrête le serveur d'un moteur ; vrai s'il tournait."""
        if not self.docker:
            return False
        entry = self.inventory(max_age=0).get(engine)
        if not entry or entry["state"] != "running":
            return False
        log.info("mise en veille de %s (%s)", engine, entry["name"])
        self.docker.stop(entry["name"])
        self.forget()
        return True

    def release(self, report: Report = log.info) -> list[str]:
        """Rend la carte : tous les serveurs qui tournent sont endormis."""
        if not self.docker:
            return []
        with self.lock:
            stopped = [e for e in self.running() if self.sleep(e)]
            for engine in stopped:
                report(f"{engine} mis en veille")
            self.touch()
            return stopped

    def maybe_release(self, busy: bool) -> list[str]:
        """Appelé à chaque battement : rend la carte passé le délai d'inactivité."""
        if not self.docker or busy:
            return []
        minutes = idle_minutes()
        if minutes <= 0 or self.idle_for() < minutes * 60:
            return []
        if not self.running():
            self.touch()
            return []
        log.info("rien à faire depuis %.0f min : la carte est rendue", minutes)
        return self.release()

    # -- ce que le battement publie ------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        minutes = idle_minutes()
        payload: dict[str, object] = {
            "docker": self.active,
            "served": self.inventory() if self.active else {},
            "idle_minutes": minutes,
        }
        if self.active and minutes > 0 and self.running():
            payload["release_in_s"] = max(0, round(minutes * 60 - self.idle_for()))
        return payload
