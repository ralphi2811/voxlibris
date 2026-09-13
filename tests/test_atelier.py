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

    def test_au_redemarrage_les_taches_en_cours_sont_remises_a_plat(self, tmp_path):
        """L'atelier qui redémarre est seul : ce qui est « en cours » ne l'est plus."""
        queue = Queue(tmp_path / "jobs.sqlite")
        stopped = queue.enqueue("livre", "proofread")
        crashed = queue.enqueue("livre", "synth")
        queue.claim()
        queue.claim()
        queue.request_cancel(stopped.id)
        assert queue.cancel_stale(older_than=0) == 1
        assert queue.get(stopped.id).state is State.CANCELLED
        assert queue.get(crashed.id).state is State.FAILED
        assert "redémarré" in queue.get(crashed.id).message
        assert queue.active("livre") is None

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
            '{"idx": 0, "text": "a"}\n{"idx": 1, "text": "b"}\n', encoding="utf-8"
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
        assert not rows[1]["text_changed"] and project.stale_chapters() == []

        # Une piste est à jour par son contenu, pas par sa date.
        (project.segments_dir / "ch01.jsonl").write_text(
            '{"idx": 0, "text": "a"}\n{"idx": 1, "text": "b corrigé"}\n', encoding="utf-8"
        )
        assert not project.tracks()[0]["current"]
        # Un texte enregistré après la préparation le dit.
        import os

        later = (project.segments_dir / "ch01.jsonl").stat().st_mtime + 60
        os.utime(project.text_path(1), (later, later))
        assert project.tracks()[0]["text_changed"] and project.stale_chapters() == [1]

        flagged = project.flagged_segments()
        assert [(f["chapter"], f["idx"], f["cause"]) for f in flagged] == [(1, 1, "trop court")]
        # Le passage autour : les voisins du manifeste.
        assert flagged[0]["before"] == "a" and flagged[0]["after"] == ""

        # Validé à l'oreille : il sort de la liste, garde sa cause, et peut y revenir.
        assert project.approve_segment(1, 1)
        assert project.flagged_segments() == []
        assert project.tracks()[0]["flagged"] == 0
        approved = project.approved_segments()
        assert [(a["idx"], a["cause"], a["clean"]) for a in approved] == [(1, "trop court", True)]
        assert project.approve_segment(1, 1, approved=False)
        assert [f["idx"] for f in project.flagged_segments()] == [1]
        assert not project.approve_segment(1, 9)
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


class TestReprise:
    def test_les_segments_inchanges_sont_repris_de_la_piste(self, tmp_path, monkeypatch):
        """Un mot corrigé ne coûte pas le chapitre : seul son segment repasse au moteur."""
        import os

        import soundfile as sf

        from voxlibris import worker
        from voxlibris.tts.quality import SAMPLE_RATE
        from voxlibris.tts.synth import LEAD_IN_MS
        from voxlibris.web.jobs import Queue

        project = make_project(tmp_path / "p")
        project.backend, project.voice = "faux", "voix"
        project.save()
        project.wav_dir.mkdir(parents=True)
        texts = ["Un deux trois.", "Quatre cinq six sept.", "Huit neuf."]
        audio = np.concatenate([np.full(SAMPLE_RATE, v, np.float32) for v in (0.1, 0.2, 0.3)])
        sf.write(project.wav_dir / "ch01.wav", audio, SAMPLE_RATE)
        timing = [
            {
                "idx": i,
                "start": float(i),
                "end": float(i + 1),
                "clean": True,
                "attempts": 1,
                "cause": "",
                "split": False,
                "text": t,
            }
            for i, t in enumerate(texts)
        ]
        (project.wav_dir / "ch01.timing.json").write_text(json.dumps(timing), encoding="utf-8")

        # Le texte du deuxième segment change ; les segments datent d'après la piste.
        texts[1] = "Quatre cinq six."
        project.segments_dir.mkdir(parents=True)
        segments = project.segments_dir / "ch01.jsonl"
        segments.write_text(
            "\n".join(
                json.dumps({"idx": i, "text": t, "pause_after_ms": 0, "chapter": 1, "title": "Un"})
                for i, t in enumerate(texts)
            )
            + "\n",
            encoding="utf-8",
        )
        later = (project.wav_dir / "ch01.wav").stat().st_mtime + 10
        os.utime(segments, (later, later))

        class FakeEngine:
            name, voice, speed, sample_rate = "faux", "voix", 1.0, SAMPLE_RATE
            supports_speed = True
            said: list[str] = []

            def say(self, text):
                self.said.append(text)
                return np.full(2 * SAMPLE_RATE, 0.9, np.float32)

        monkeypatch.setattr("voxlibris.tts.backends.load", lambda *a, **k: FakeEngine())
        monkeypatch.setattr(
            "voxlibris.tts.synth.profile_for_voice",
            lambda *a, **k: __import__(
                "voxlibris.tts.quality", fromlist=["QualityProfile"]
            ).QualityProfile(chars_per_second=10),
        )
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("p", "synth", backend="faux")
        worker.run_synth(project, job, queue)

        assert FakeEngine.said == ["Quatre cinq six."]
        rebuilt, _ = sf.read(project.wav_dir / "ch01.wav", dtype="float32")
        lead = SAMPLE_RATE * LEAD_IN_MS // 1000  # la piste ouvre sur un silence
        assert len(rebuilt) == 4 * SAMPLE_RATE + lead
        assert rebuilt[lead + SAMPLE_RATE // 2] == pytest.approx(0.1, abs=1e-3)
        assert rebuilt[lead + SAMPLE_RATE + 10] == pytest.approx(0.9, abs=1e-3)
        assert rebuilt[lead + 3 * SAMPLE_RATE + 10] == pytest.approx(0.3, abs=1e-3)
        after = project.timing(1)
        assert [e["reused"] for e in after] == [True, False, True]
        assert after[2]["start"] == pytest.approx(3.0 + LEAD_IN_MS / 1000, abs=0.05)
        assert "2 segments repris" in queue.get(job.id).log

    def test_le_texte_corrige_est_reprepare_avant_la_synthese(self, tmp_path, monkeypatch):
        """Corriger puis lancer la synthèse suffit : la préparation est refaite d'abord."""
        import os

        import soundfile as sf

        from voxlibris import worker
        from voxlibris.tts.quality import SAMPLE_RATE
        from voxlibris.web.jobs import Queue

        project = make_project(tmp_path / "p")
        project.backend, project.voice = "faux", "voix"
        project.save()
        project.segments_dir.mkdir(parents=True)
        segments = project.segments_dir / "ch01.jsonl"
        segments.write_text(
            json.dumps({"idx": 0, "text": "Un.", "pause_after_ms": 0, "chapter": 1, "title": "Un"})
            + "\n",
            encoding="utf-8",
        )
        project.wav_dir.mkdir(parents=True)
        sf.write(project.wav_dir / "ch01.wav", np.full(SAMPLE_RATE, 0.1, np.float32), SAMPLE_RATE)
        (project.wav_dir / "ch01.timing.json").write_text(
            json.dumps(
                [
                    {
                        "idx": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "clean": True,
                        "attempts": 1,
                        "cause": "",
                        "split": False,
                        "text": "Un.",
                    }
                ]
            ),
            encoding="utf-8",
        )
        later = segments.stat().st_mtime + 60
        os.utime(project.text_path(1), (later, later))

        prepared = []
        monkeypatch.setattr(
            "voxlibris.worker.build_segments", lambda *a, **k: prepared.append(a) or {1: 1}
        )

        class FakeEngine:
            name, voice, speed, sample_rate = "faux", "voix", 1.0, SAMPLE_RATE
            supports_speed = True

            def say(self, text):
                return np.full(SAMPLE_RATE, 0.9, np.float32)

        monkeypatch.setattr("voxlibris.tts.backends.load", lambda *a, **k: FakeEngine())
        monkeypatch.setattr(
            "voxlibris.tts.synth.profile_for_voice",
            lambda *a, **k: __import__(
                "voxlibris.tts.quality", fromlist=["QualityProfile"]
            ).QualityProfile(chars_per_second=10),
        )
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("p", "synth", backend="faux")
        worker.run_synth(project, job, queue)
        assert prepared and prepared[0][1] == project.segments_dir
        assert "repréparés d'abord" in queue.get(job.id).log
        # Les segments n'ayant pas changé, la piste est reconnue à jour.
        assert "déjà synthétisé" in queue.get(job.id).log


class TestDistribution:
    """Deux voix sur un même moteur, et ce que coûte d'en changer une."""

    @staticmethod
    def make(tmp_path, monkeypatch):
        from voxlibris.normalize import build_segments
        from voxlibris.tts.quality import SAMPLE_RATE

        project = make_project(tmp_path / "p")
        (project.raw_dir / "ch01.md").write_text(
            '---\nchapter: 1\ntitle: "Un"\n---\n\nLe soir tombait sur la ferme.\n\n'
            "— On part maintenant, souffla Lulu à son frère.\n",
            encoding="utf-8",
        )
        (project.raw_dir / "ch02.md").unlink()
        project.backend, project.voice, project.announce_chapters = "faux", "voix", False
        project.save()
        build_segments(project.raw_dir, project.segments_dir, announce_chapters=False)

        class FakeEngine:
            name, speed, sample_rate, supports_speed = "faux", 1.0, SAMPLE_RATE, True
            said: list[tuple[str, str]] = []

            def __init__(self, voice="voix"):
                self.voice = voice

            def voices(self):
                return ["voix", "autre", "tierce"]

            def cast(self, voice):
                assert voice in self.voices()
                return FakeEngine(voice)

            def say(self, text):
                # Une durée au débit calibré, sans quoi le contrôle qualité rejoue tout.
                FakeEngine.said.append((self.voice, text))
                return np.full(SAMPLE_RATE * len(text) // 10, 0.9, np.float32)

        monkeypatch.setattr("voxlibris.tts.backends.load", lambda *a, **k: FakeEngine())
        monkeypatch.setattr(
            "voxlibris.tts.synth.profile_for_voice",
            lambda *a, **k: __import__(
                "voxlibris.tts.quality", fromlist=["QualityProfile"]
            ).QualityProfile(chars_per_second=10),
        )
        return project, FakeEngine

    def test_les_repliques_passent_par_la_voix_des_dialogues(self, tmp_path, monkeypatch):
        from voxlibris import worker
        from voxlibris.project import Project, read_stamp

        project, engine = self.make(tmp_path, monkeypatch)
        queue = Queue(tmp_path / "jobs.sqlite")
        project.voices = {"dialogue": "autre"}
        project.save()
        job = queue.enqueue("p", "synth", backend="faux")
        worker.run_synth(project, job, queue)

        assert engine.said == [
            ("voix", "Le soir tombait sur la ferme."),
            ("autre", "On part maintenant, souffla Lulu à son frère."),
        ]
        assert read_stamp(project.wav_dir / "ch01.wav") == "faux/voix+dialogue=autre@1.00"
        assert Project.load(project.root).voices == {"dialogue": "autre"}
        assert [(e["role"], e["voice"]) for e in project.timing(1)] == [
            ("narrateur", "voix"),
            ("dialogue", "autre"),
        ]
        assert "dialogue par autre (1 segments)" in queue.get(job.id).log

    def test_changer_une_voix_ne_refait_que_ce_qu_elle_disait(self, tmp_path, monkeypatch):
        from voxlibris import worker
        from voxlibris.project import read_stamp

        project, engine = self.make(tmp_path, monkeypatch)
        queue = Queue(tmp_path / "jobs.sqlite")
        project.voices = {"dialogue": "autre"}
        worker.run_synth(project, queue.enqueue("p", "synth", backend="faux"), queue)

        engine.said.clear()
        project.voices = {"dialogue": "tierce"}
        job = queue.enqueue("p", "synth", backend="faux")
        worker.run_synth(project, job, queue)
        assert engine.said == [("tierce", "On part maintenant, souffla Lulu à son frère.")]
        assert [e["reused"] for e in project.timing(1)] == [True, False]
        assert "ch01 : voix changée" in queue.get(job.id).log
        assert read_stamp(project.wav_dir / "ch01.wav") == "faux/voix+dialogue=tierce@1.00"

        # Retour à une seule voix : le narrateur reprend les répliques, et rien d'autre.
        engine.said.clear()
        project.voices = {}
        worker.run_synth(project, queue.enqueue("p", "synth", backend="faux"), queue)
        assert engine.said == [("voix", "On part maintenant, souffla Lulu à son frère.")]
        assert read_stamp(project.wav_dir / "ch01.wav") == "faux/voix@1.00"

    def test_un_persona_du_texte_a_sa_voix(self, tmp_path, monkeypatch):
        """Les lettres de Charles, marquées dans le texte, passent par la voix de Charles."""
        from voxlibris import worker
        from voxlibris.normalize import build_segments
        from voxlibris.project import read_stamp

        project, engine = self.make(tmp_path, monkeypatch)
        (project.raw_dir / "ch01.md").write_text(
            '---\nchapter: 1\ntitle: "Un"\n---\n\nLe facteur a apporté une lettre.\n\n'
            "@Charles\n\nMa Lulu, nous n'avons pas arrêté de bouger.\n\n"
            "— Rassemblement ! a crié le lieutenant.\n\n@\n\n"
            "— Il va bien, ai-je dit à maman.\n",
            encoding="utf-8",
        )
        build_segments(project.raw_dir, project.segments_dir, announce_chapters=False)
        project.voices = {"charles": "autre", "Inconnu": "tierce"}
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("p", "synth", backend="faux")
        worker.run_synth(project, job, queue)

        assert [v for v, _ in engine.said] == ["voix", "autre", "autre", "voix"]
        assert read_stamp(project.wav_dir / "ch01.wav") == "faux/voix+charles=autre@1.00"
        log = queue.get(job.id).log
        assert "charles par autre (2 segments)" in log
        assert "Inconnu : aucun paragraphe attribué, sa voix ne servira pas" in log
        # Le projet juge la piste comme la synthèse l'a notée : ni la casse du nom, ni
        # une voix qui n'a pas servi ne la font passer pour périmée.
        assert project.signature == read_stamp(project.wav_dir / "ch01.wav")
        assert project.track_matches(1)

    def test_un_persona_sans_voix_ne_change_rien(self, tmp_path, monkeypatch):
        """Un persona nommé sans voix : le narrateur lit tout, la note ne le mentionne pas."""
        from voxlibris import worker
        from voxlibris.project import read_stamp

        project, engine = self.make(tmp_path, monkeypatch)
        project.voices = {"Charles": ""}
        project.save()
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("p", "synth", backend="faux")
        worker.run_synth(project, job, queue)
        assert [v for v, _ in engine.said] == ["voix", "voix"]
        assert read_stamp(project.wav_dir / "ch01.wav") == "faux/voix@1.00"


class TestSignature:
    def test_la_note_se_relit(self):
        from voxlibris.project import make_signature, parse_signature, same_engine

        alone = make_signature("xtts", "Viktor Menelaos", 0.95)
        both = make_signature(
            "xtts", "Viktor Menelaos", 0.95, {"Charles": "Damien", "dialogue": "Ana", "x": ""}
        )
        assert alone == "xtts/Viktor Menelaos@0.95"
        assert both == "xtts/Viktor Menelaos+charles=Damien+dialogue=Ana@0.95"
        assert parse_signature(both) == (
            "xtts",
            "Viktor Menelaos",
            {"charles": "Damien", "dialogue": "Ana"},
            "0.95",
        )
        assert parse_signature(alone) == ("xtts", "Viktor Menelaos", {}, "0.95")
        # Les premières notes à deux voix n'écrivaient que la voix des dialogues.
        assert parse_signature("xtts/Viktor+Ana@1.00")[2] == {"dialogue": "Ana"}
        assert same_engine(alone, both)
        assert not same_engine(alone, make_signature("xtts", "Viktor Menelaos", 1.0))
        assert not same_engine(alone, make_signature("kokoro", "Viktor Menelaos", 0.95))
        assert not same_engine("", alone)
