"""L'Atelier : réglages superposés, arrêt d'une tâche, battement, pistes et segments signalés.

Ce qui se teste ici est ce qui relie l'interface à l'atelier sans qu'ils se parlent :
un fichier de réglages relu à chaque accès, une colonne dans la file, un battement sur
disque. Chacun de ces liens, cassé en silence, laisserait l'interface mentir.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from voxlibris import atelier, config
from voxlibris.project import Project
from voxlibris.web.jobs import Cancelled, Queue, State


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXLIBRIS_DATA", str(tmp_path))
    # Le cache des réglages est indexé sur la date du fichier : on repart de zéro.
    config._settings_cache = (-1.0, {})
    return tmp_path


class TestReglages:
    def test_l_interface_prime_sur_l_environnement(self, data, monkeypatch):
        monkeypatch.setenv("VOXLIBRIS_LLM_MODEL", "depuis-env")
        assert config.setting("VOXLIBRIS_LLM_MODEL") == "depuis-env"
        assert config.origin("VOXLIBRIS_LLM_MODEL") == "environnement"

        config.write_settings({"VOXLIBRIS_LLM_MODEL": "depuis-interface"})
        assert config.setting("VOXLIBRIS_LLM_MODEL") == "depuis-interface"
        assert config.origin("VOXLIBRIS_LLM_MODEL") == "interface"

    def test_une_valeur_vide_efface_le_reglage(self, data):
        config.write_settings({"VOXLIBRIS_MISTRAL_BASE_URL": "http://voxtral:8600/v1"})
        assert config.setting("VOXLIBRIS_MISTRAL_BASE_URL") == "http://voxtral:8600/v1"
        config.write_settings({"VOXLIBRIS_MISTRAL_BASE_URL": ""})
        assert "VOXLIBRIS_MISTRAL_BASE_URL" not in config.read_settings()

    def test_seules_les_cles_connues_sont_ecrites(self, data):
        config.write_settings({"PATH": "/tmp", "VOXLIBRIS_DEVICE": "cpu"})
        assert config.read_settings() == {"VOXLIBRIS_DEVICE": "cpu"}

    def test_un_fichier_modifie_a_la_main_est_relu(self, data):
        """L'atelier et l'interface partagent le fichier : l'un doit voir ce que l'autre écrit."""
        config.write_settings({"VOXLIBRIS_DEVICE": "cpu"})
        assert config.setting("VOXLIBRIS_DEVICE") == "cpu"
        path = config.settings_file()
        path.write_text(json.dumps({"VOXLIBRIS_DEVICE": "cuda"}), encoding="utf-8")
        # Une écriture dans la même seconde garderait la même date : on la force.
        later = path.stat().st_mtime + 2
        import os

        os.utime(path, (later, later))
        assert config.setting("VOXLIBRIS_DEVICE") == "cuda"


class TestArret:
    def test_une_tache_en_attente_s_arrete_sur_le_champ(self, tmp_path):
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("livre", "synth")
        assert queue.request_cancel(job.id)
        assert queue.get(job.id).state is State.CANCELLED
        assert queue.claim() is None

    def test_une_tache_en_cours_est_signalee_puis_arretee(self, tmp_path):
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("livre", "synth")
        queue.claim()
        assert not queue.cancel_requested(job.id)
        assert queue.request_cancel(job.id)
        assert queue.cancel_requested(job.id)
        # L'atelier relève la demande entre deux segments et termine proprement.
        queue.finish(job.id, Cancelled())
        done = queue.get(job.id)
        assert done.state is State.CANCELLED
        assert "arrêtée" in done.message
        assert "Traceback" not in done.log

    def test_une_base_ancienne_recoit_la_colonne(self, tmp_path):
        """Une base créée avant l'arrêt à la demande doit continuer de fonctionner."""
        import sqlite3

        path = tmp_path / "jobs.sqlite"
        db = sqlite3.connect(path)
        db.executescript(
            """CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT NOT NULL,
            kind TEXT NOT NULL, params TEXT NOT NULL DEFAULT '{}', state TEXT NOT NULL
            DEFAULT 'en attente', progress REAL NOT NULL DEFAULT 0.0, message TEXT NOT NULL
            DEFAULT '', log TEXT NOT NULL DEFAULT '', created REAL NOT NULL, started REAL,
            finished REAL);
            INSERT INTO jobs (project, kind, created) VALUES ('livre', 'synth', 1.0);"""
        )
        db.close()
        queue = Queue(path)
        assert queue.get(1).cancel == 0
        assert queue.request_cancel(1)


class TestBattement:
    def test_absent_sans_fichier(self, tmp_path):
        assert atelier.status(tmp_path)["online"] is False

    def test_present_puis_perime(self, tmp_path, monkeypatch):
        monkeypatch.setattr(atelier, "engines", lambda: {"piper": True, "xtts": False})
        monkeypatch.setattr(atelier, "hardware", lambda: {"device": "cpu", "gpu": ""})
        atelier.beat(tmp_path, busy=7)
        seen = atelier.status(tmp_path)
        assert seen["online"] and seen["busy"] == 7 and seen["engines"]["piper"]

        stale = json.loads(atelier.heartbeat_file(tmp_path).read_text())
        stale["time"] = time.time() - atelier.STALE - 1
        atelier.heartbeat_file(tmp_path).write_text(json.dumps(stale))
        assert atelier.status(tmp_path)["online"] is False

    def test_un_fichier_illisible_vaut_absent(self, tmp_path):
        atelier.heartbeat_file(tmp_path).write_text("{pas du json")
        assert atelier.status(tmp_path)["online"] is False


def make_project(root) -> Project:
    project = Project(root=root, title="Essai")
    project.raw_dir.mkdir(parents=True)
    for number, title in ((1, "Un"), (2, "Deux")):
        (project.raw_dir / f"ch{number:02d}.md").write_text(
            f'---\nchapter: {number}\ntitle: "{title}"\n---\n\nIl faisait beau. Puis il plut.\n',
            encoding="utf-8",
        )
    project.save()
    return project


class TestPistes:
    def test_renommer_un_chapitre(self, tmp_path):
        project = make_project(tmp_path / "p")
        project.clean_dir.mkdir(parents=True)
        (project.clean_dir / "ch01.md").write_text(
            (project.raw_dir / "ch01.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
        project.set_chapter_title(1, "  Premier matin ")
        titles = {s["number"]: s["title"] for s in project.chapter_states()}
        assert titles[1] == "Premier matin"
        assert 'title: "Premier matin"' in (project.clean_dir / "ch01.md").read_text()
        with pytest.raises(ValueError):
            project.set_chapter_title(1, "   ")
        with pytest.raises(FileNotFoundError):
            project.set_chapter_title(9, "Neuf")

    def test_pistes_et_segments_signales(self, tmp_path):
        project = make_project(tmp_path / "p")
        project.segments_dir.mkdir(parents=True)
        (project.segments_dir / "ch01.jsonl").write_text(
            '{"idx": 0}\n{"idx": 1}\n', encoding="utf-8"
        )
        project.wav_dir.mkdir(parents=True)
        (project.wav_dir / "ch01.wav").write_bytes(b"RIFF")
        (project.wav_dir / "ch01.timing.json").write_text(
            json.dumps(
                [
                    {"idx": 0, "start": 0.0, "end": 2.0, "clean": True, "attempts": 1, "text": "a"},
                    {
                        "idx": 1,
                        "start": 2.5,
                        "end": 3.0,
                        "clean": False,
                        "cause": "trop court",
                        "attempts": 3,
                        "text": "b",
                    },
                ]
            ),
            encoding="utf-8",
        )
        rows = {r["number"]: r for r in project.tracks()}
        assert rows[1]["segments"] == 2 and rows[1]["synthesized"] and rows[1]["current"]
        assert rows[1]["seconds"] == 3.0 and rows[1]["flagged"] == 1
        assert not rows[2]["synthesized"] and rows[2]["segments"] == 0

        flagged = project.flagged_segments()
        assert [(f["chapter"], f["idx"], f["cause"]) for f in flagged] == [(1, 1, "trop court")]
        assert project.audio_seconds() == 3.0


class TestRejeu:
    def test_le_segment_est_recolle_et_la_suite_decalee(self, tmp_path, monkeypatch):
        """Rejouer un segment remplace sa plage dans la piste et décale ce qui suit."""
        import soundfile as sf

        from voxlibris import worker
        from voxlibris.tts.quality import SAMPLE_RATE
        from voxlibris.web.jobs import Queue

        project = make_project(tmp_path / "p")
        project.backend, project.voice = "faux", "voix"
        project.save()
        project.segments_dir.mkdir(parents=True)
        (project.segments_dir / "ch01.jsonl").write_text(
            "\n".join(
                json.dumps({"idx": i, "text": t, "pause_after_ms": 0, "chapter": 1, "title": "Un"})
                for i, t in enumerate(["Un deux trois.", "Quatre cinq six sept.", "Huit neuf."])
            )
            + "\n",
            encoding="utf-8",
        )
        project.wav_dir.mkdir(parents=True)
        # Trois segments d'une seconde chacun, marqués 1, 2, 3 pour les reconnaître.
        audio = np.concatenate([np.full(SAMPLE_RATE, v, np.float32) for v in (0.1, 0.2, 0.3)])
        sf.write(project.wav_dir / "ch01.wav", audio, SAMPLE_RATE)
        timing = [
            {
                "idx": i,
                "start": float(i),
                "end": float(i + 1),
                "clean": i != 1,
                "cause": "trop court" if i == 1 else "",
                "attempts": 1,
                "split": False,
                "text": "x",
            }
            for i in range(3)
        ]
        (project.wav_dir / "ch01.timing.json").write_text(json.dumps(timing), encoding="utf-8")

        class FakeEngine:
            name, voice, speed, sample_rate = "faux", "voix", 1.0, SAMPLE_RATE
            supports_speed = True

            def say(self, text):  # deux secondes, valeur 0.9 : plus long que l'original
                return np.full(2 * SAMPLE_RATE, 0.9, np.float32)

        monkeypatch.setattr("voxlibris.tts.backends.load", lambda *a, **k: FakeEngine())
        monkeypatch.setattr(
            "voxlibris.tts.synth.profile_for_voice",
            lambda *a, **k: __import__(
                "voxlibris.tts.quality", fromlist=["QualityProfile"]
            ).QualityProfile(chars_per_second=10),
        )
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("p", "resynth", chapter=1, idx=1)
        worker.run_resynth(project, job, queue)

        patched, _ = sf.read(project.wav_dir / "ch01.wav", dtype="float32")
        assert len(patched) == 4 * SAMPLE_RATE
        assert patched[SAMPLE_RATE // 2] == pytest.approx(0.1, abs=1e-3)
        assert patched[SAMPLE_RATE + 10] == pytest.approx(0.9, abs=1e-3)
        assert patched[3 * SAMPLE_RATE + 10] == pytest.approx(0.3, abs=1e-3)
        after = project.timing(1)
        assert after[1]["start"] == 1.0 and after[1]["end"] == pytest.approx(3.0, abs=0.05)
        assert after[2]["start"] == pytest.approx(3.0, abs=0.05)
