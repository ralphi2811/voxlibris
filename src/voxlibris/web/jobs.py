"""File de tâches adossée à SQLite.

Une synthèse dure des dizaines de minutes : elle ne peut pas se dérouler dans une requête
HTTP. Il faut donc une file, et un atelier qui la consomme dans un autre processus — ce
qui permet au passage de n'installer CUDA que de son côté, l'interface web n'en ayant
aucun besoin.

Le choix de SQLite plutôt que Redis mérite d'être expliqué, car il va à rebours de
l'habitude. Pour un outil auto-hébergé mono-utilisateur, une file en base supprime un
service à déployer et rend l'interface lançable d'un simple `uvicorn`, sans rien d'autre
à démarrer. La progression est relue par sondage plutôt que reçue par abonnement, mais
sur une tâche de vingt minutes un rafraîchissement à la seconde est amplement suffisant.
Redis redeviendrait justifié s'il fallait plusieurs ateliers concurrents.
"""

from __future__ import annotations

import json
import sqlite3
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class State(StrEnum):
    PENDING = "en attente"
    RUNNING = "en cours"
    DONE = "terminé"
    FAILED = "échoué"
    CANCELLED = "arrêtée"


class Cancelled(Exception):
    """Levée par l'atelier quand l'interface a demandé l'arrêt de la tâche en cours."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    project   TEXT NOT NULL,
    kind      TEXT NOT NULL,
    params    TEXT NOT NULL DEFAULT '{}',
    state     TEXT NOT NULL DEFAULT 'en attente',
    progress  REAL NOT NULL DEFAULT 0.0,
    message   TEXT NOT NULL DEFAULT '',
    log       TEXT NOT NULL DEFAULT '',
    created   REAL NOT NULL,
    started   REAL,
    finished  REAL,
    cancel    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_pending ON jobs(state, id);
"""


@dataclass
class Job:
    id: int
    project: str
    kind: str
    params: dict
    state: State
    progress: float
    message: str
    log: str
    created: float
    started: float | None
    finished: float | None
    cancel: int = 0

    @property
    def running_for(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started


class Queue:
    """File de tâches persistante."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)
            # Une base créée par une version antérieure n'a pas la colonne d'arrêt.
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            if "cancel" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN cancel INTEGER NOT NULL DEFAULT 0")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        # WAL : l'atelier écrit sa progression pendant que l'interface la lit, sans
        # qu'aucun des deux n'attende l'autre.
        db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    def _row(self, row: sqlite3.Row) -> Job:
        data = dict(row)
        data["params"] = json.loads(data["params"])
        data["state"] = State(data["state"])
        return Job(**data)

    def enqueue(self, project: str, kind: str, **params) -> Job:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO jobs (project, kind, params, created) VALUES (?, ?, ?, ?)",
                (project, kind, json.dumps(params), time.time()),
            )
            return self.get(cursor.lastrowid)  # type: ignore[arg-type]

    def get(self, job_id: int) -> Job:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"tâche {job_id} inconnue")
        return self._row(row)

    def list(self, project: str | None = None, limit: int = 50) -> list[Job]:
        query = "SELECT * FROM jobs"
        args: tuple = ()
        if project:
            query += " WHERE project = ?"
            args = (project,)
        query += " ORDER BY id DESC LIMIT ?"
        with self._connect() as db:
            rows = db.execute(query, (*args, limit)).fetchall()
        return [self._row(row) for row in rows]

    def active(self, project: str) -> Job | None:
        """Tâche en attente ou en cours pour ce projet, s'il y en a une."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE project = ? AND state IN (?, ?) ORDER BY id LIMIT 1",
                (project, State.PENDING, State.RUNNING),
            ).fetchone()
        return self._row(row) if row else None

    def claim(self) -> Job | None:
        """Prend la plus ancienne tâche en attente et la marque en cours.

        La mise à jour conditionnelle fait office de verrou : si deux ateliers tentent la
        même tâche, un seul verra `rowcount` valoir 1.
        """
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE state = ? ORDER BY id LIMIT 1", (State.PENDING,)
            ).fetchone()
            if row is None:
                return None
            cursor = db.execute(
                "UPDATE jobs SET state = ?, started = ? WHERE id = ? AND state = ?",
                (State.RUNNING, time.time(), row["id"], State.PENDING),
            )
            if cursor.rowcount != 1:
                return None
        return self.get(row["id"])

    def report(self, job_id: int, progress: float | None = None, message: str = "") -> None:
        with self._connect() as db:
            if progress is not None:
                db.execute("UPDATE jobs SET progress = ? WHERE id = ?", (progress, job_id))
            if message:
                db.execute(
                    "UPDATE jobs SET message = ?, log = substr(log || ? , -20000) WHERE id = ?",
                    (message, message + "\n", job_id),
                )

    def request_cancel(self, job_id: int) -> bool:
        """Demande l'arrêt d'une tâche. Une tâche encore en attente est arrêtée sur-le-champ.

        Une tâche en cours, elle, ne peut être interrompue qu'entre deux segments : c'est
        l'atelier qui relève la demande et s'arrête proprement, sans piste à moitié écrite.
        """
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET state = ?, finished = ?, message = ? WHERE id = ? AND state = ?",
                (State.CANCELLED, time.time(), "arrêtée avant de commencer", job_id, State.PENDING),
            )
            if cursor.rowcount == 1:
                return True
            cursor = db.execute(
                "UPDATE jobs SET cancel = 1, message = ? WHERE id = ? AND state = ?",
                ("arrêt demandé…", job_id, State.RUNNING),
            )
            return cursor.rowcount == 1

    def cancel_requested(self, job_id: int) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT cancel FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return bool(row and row["cancel"])

    def finish(self, job_id: int, error: BaseException | None = None) -> None:
        if isinstance(error, Cancelled):
            with self._connect() as db:
                db.execute(
                    "UPDATE jobs SET state = ?, finished = ?, message = ? WHERE id = ?",
                    (State.CANCELLED, time.time(), "arrêtée à la demande", job_id),
                )
            return
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET state = ?, finished = ?, progress = ?, message = ? WHERE id = ?",
                (
                    State.FAILED if error else State.DONE,
                    time.time(),
                    1.0,
                    _describe(error) if error else "terminé",
                    job_id,
                ),
            )
            if error:
                db.execute(
                    "UPDATE jobs SET log = log || ? WHERE id = ?",
                    ("\n" + "".join(traceback.format_exception(error)), job_id),
                )

    def cancel_stale(self, older_than: float = 6 * 3600) -> int:
        """Marque en échec les tâches restées en cours après un arrêt brutal."""
        cutoff = time.time() - older_than
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET state = ?, message = ? WHERE state = ? AND started < ?",
                (State.FAILED, "interrompue", State.RUNNING, cutoff),
            )
            return cursor.rowcount


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def default_queue() -> Queue:
    return Queue(workspace() / "jobs.sqlite")


def workspace() -> Path:
    """Dossier où vivent les projets et la file."""
    from ..config import data_dir

    return data_dir()
