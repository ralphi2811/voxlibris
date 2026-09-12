"""Tests de l'interface web et de la file de tâches.

L'application ne fait aucun travail lourd : elle dépose des tâches et affiche leur
avancement. C'est donc cela qu'on vérifie ici — le parcours et la file — sans jamais
charger de moteur de synthèse.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from voxlibris.web.jobs import Queue, State


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Application isolée dans un espace de travail temporaire."""
    monkeypatch.setenv("VOXLIBRIS_DATA", str(tmp_path / "data"))
    # Les modules retiennent l'espace de travail à l'import : il faut les recharger.
    import importlib

    from voxlibris.web import app as app_module
    from voxlibris.web import jobs as jobs_module

    importlib.reload(jobs_module)
    importlib.reload(app_module)
    return TestClient(app_module.app)


class TestQueue:
    def test_depot_et_relecture(self, tmp_path):
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("livre", "synth", backend="piper")
        assert queue.get(job.id).params["backend"] == "piper"
        assert queue.get(job.id).state is State.PENDING

    def test_une_tache_nest_prise_quune_fois(self, tmp_path):
        """La prise en charge fait office de verrou entre ateliers concurrents."""
        queue = Queue(tmp_path / "jobs.sqlite")
        queue.enqueue("livre", "normalize")
        assert queue.claim() is not None
        assert queue.claim() is None

    def test_ordre_darrivee(self, tmp_path):
        queue = Queue(tmp_path / "jobs.sqlite")
        queue.enqueue("a", "normalize")
        queue.enqueue("b", "normalize")
        assert queue.claim().project == "a"
        assert queue.claim().project == "b"

    def test_progression_et_journal(self, tmp_path):
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("livre", "synth")
        queue.report(job.id, 0.5, "chapitre 3")
        refreshed = queue.get(job.id)
        assert refreshed.progress == 0.5
        assert "chapitre 3" in refreshed.log

    def test_echec_conserve_la_trace(self, tmp_path):
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("livre", "synth")
        queue.finish(job.id, RuntimeError("moteur absent"))
        failed = queue.get(job.id)
        assert failed.state is State.FAILED
        assert "moteur absent" in failed.message

    def test_tache_active_par_projet(self, tmp_path):
        queue = Queue(tmp_path / "jobs.sqlite")
        queue.enqueue("livre-a", "normalize")
        assert queue.active("livre-a") is not None
        assert queue.active("livre-b") is None

    def test_taches_interrompues_remises_a_plat(self, tmp_path):
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("livre", "synth")
        queue.claim()
        assert queue.cancel_stale(older_than=-1) == 1
        assert queue.get(job.id).state is State.FAILED


class TestParcours:
    def test_accueil_vide(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Aucun livre" in response.text

    def test_depot_dun_epub(self, client, make_epub):
        path = make_epub(["Le départ", "La traversée"])
        with path.open("rb") as handle:
            response = client.post("/projects", files={"file": (path.name, handle)})
        assert response.status_code == 200  # redirection suivie
        assert "Le registre du gardien" in response.text
        # Un EPUB n'a rien à relire.
        assert "relecture nécessaire" not in response.text

    def test_le_fichier_se_presente(self, client, make_epub):
        """Avant l'import, le formulaire demande au livre son titre et sa langue."""
        path = make_epub(["Le départ"])
        with path.open("rb") as handle:
            response = client.post("/peek", files={"file": (path.name, handle)})
        assert response.status_code == 200
        assert response.json()["title"] == "Le registre du gardien"
        assert response.json()["language"] == "fr"

    def test_format_refuse_proprement(self, client, tmp_path):
        path = tmp_path / "livre.docx"
        path.write_bytes(b"peu importe")
        with path.open("rb") as handle:
            response = client.post("/projects", files={"file": (path.name, handle)})
        assert response.status_code == 400

    def test_projet_inconnu(self, client):
        assert client.get("/projects/fantome").status_code == 404

    def test_evasion_de_chemin_refusee(self, client):
        assert client.get("/projects/..%2F..%2Fetc").status_code == 404


class TestSuppression:
    def _create(self, client, make_epub):
        path = make_epub(["Le départ", "La traversée"])
        with path.open("rb") as handle:
            client.post("/projects", files={"file": (path.name, handle)})
        return "le-registre-du-gardien"

    def test_supprime_et_revient_a_la_liste(self, client, make_epub):
        name = self._create(client, make_epub)
        response = client.post(f"/projects/{name}/delete")
        assert response.status_code == 200  # redirection suivie
        assert "Aucun livre" in response.text
        assert client.get(f"/projects/{name}").status_code == 404

    def test_refuse_pendant_une_tache(self, client, make_epub):
        """Supprimer sous les pieds de l'atelier le ferait écrire dans le vide."""
        name = self._create(client, make_epub)
        client.post(f"/projects/{name}/jobs/normalize")

        assert client.post(f"/projects/{name}/delete").status_code == 409
        assert client.get(f"/projects/{name}").status_code == 200

    def test_projet_inconnu(self, client):
        assert client.post("/projects/fantome/delete").status_code == 404

    def test_evasion_de_chemin_refusee(self, client):
        assert client.post("/projects/..%2F..%2Fetc/delete").status_code == 404


class TestChapitres:
    def _create(self, client, make_epub):
        path = make_epub(["Avant-propos", "Le départ", "La traversée"])
        with path.open("rb") as handle:
            client.post("/projects", files={"file": (path.name, handle)})
        return "le-registre-du-gardien"

    def _titles(self, client, name):
        from voxlibris.project import Project
        from voxlibris.web.jobs import workspace

        return [s["title"] for s in Project.load(workspace() / name).chapter_states()]

    def test_retire_et_renumerote(self, client, make_epub):
        """Une page de copyright en tête décale tout le livre : il faut l'ôter."""
        name = self._create(client, make_epub)
        assert self._titles(client, name) == ["Avant-propos", "Le départ", "La traversée"]

        response = client.post(f"/projects/{name}/chapters/1/delete")
        assert response.status_code == 200  # redirection suivie
        assert self._titles(client, name) == ["Le départ", "La traversée"]

    def test_le_texte_relu_suit_la_renumerotation(self, client, make_epub, tmp_path):
        name = self._create(client, make_epub)
        client.post(f"/projects/{name}/chapters/1/delete")

        clean = tmp_path / "data" / name / "text" / "clean"
        assert sorted(p.name for p in clean.glob("ch*.md")) == ["ch01.md", "ch02.md"]
        assert "chapter: 1" in (clean / "ch01.md").read_text(encoding="utf-8")

    def test_recolle_au_precedent(self, client, make_epub, tmp_path):
        """Une illustration coupe un chapitre en deux : le fragment se rattache."""
        name = self._create(client, make_epub)
        clean = tmp_path / "data" / name / "text" / "clean"
        avant = (clean / "ch01.md").read_text(encoding="utf-8")
        suite = (clean / "ch02.md").read_text(encoding="utf-8").split("---", 2)[-1].strip()

        assert client.post(f"/projects/{name}/chapters/2/merge").status_code == 200
        assert self._titles(client, name) == ["Avant-propos", "La traversée"]

        fusionne = (clean / "ch01.md").read_text(encoding="utf-8")
        assert avant.split("---", 2)[-1].strip() in fusionne
        assert suite in fusionne

    def test_le_premier_chapitre_na_pas_de_precedent(self, client, make_epub):
        name = self._create(client, make_epub)
        assert client.post(f"/projects/{name}/chapters/1/merge").status_code == 400

    def test_chapitre_inconnu(self, client, make_epub):
        name = self._create(client, make_epub)
        assert client.post(f"/projects/{name}/chapters/9/delete").status_code == 404

    def test_action_inconnue(self, client, make_epub):
        name = self._create(client, make_epub)
        assert client.post(f"/projects/{name}/chapters/1/renommer").status_code == 404

    def test_refuse_pendant_une_tache(self, client, make_epub):
        name = self._create(client, make_epub)
        client.post(f"/projects/{name}/jobs/normalize")
        assert client.post(f"/projects/{name}/chapters/1/delete").status_code == 409


class TestRelecture:
    def _create(self, client, make_epub):
        path = make_epub(["Le départ", "La traversée"])
        with path.open("rb") as handle:
            client.post("/projects", files={"file": (path.name, handle)})
        return "le-registre-du-gardien"

    def test_ouvre_lediteur(self, client, make_epub):
        name = self._create(client, make_epub)
        response = client.get(f"/projects/{name}/review/1")
        assert response.status_code == 200
        assert "Le départ" in response.text

    def test_enregistre_sans_toucher_a_lextraction(self, client, make_epub, tmp_path):
        name = self._create(client, make_epub)
        client.post(
            f"/projects/{name}/review/1",
            data={"title": "Titre corrigé", "text": "Un paragraphe.\n\nUn autre."},
        )
        root = tmp_path / "data" / name
        clean = (root / "text" / "clean" / "ch01.md").read_text(encoding="utf-8")
        raw = (root / "text" / "raw" / "ch01.md").read_text(encoding="utf-8")
        assert "Titre corrigé" in clean
        assert "Titre corrigé" not in raw

    def test_chapitre_inconnu(self, client, make_epub):
        name = self._create(client, make_epub)
        assert client.get(f"/projects/{name}/review/99").status_code == 404


class TestTaches:
    def _create(self, client, make_epub):
        path = make_epub(["Le départ"])
        with path.open("rb") as handle:
            client.post("/projects", files={"file": (path.name, handle)})
        return "le-registre-du-gardien"

    def test_depot_dune_tache(self, client, make_epub):
        name = self._create(client, make_epub)
        response = client.post(f"/projects/{name}/jobs/normalize")
        assert response.status_code == 200
        assert "normalize" in client.get(f"/projects/{name}/progress").text

    def test_une_seule_tache_a_la_fois(self, client, make_epub):
        name = self._create(client, make_epub)
        client.post(f"/projects/{name}/jobs/normalize")
        second = client.post(f"/projects/{name}/jobs/normalize", follow_redirects=False)
        assert second.status_code == 409


class TestVoxtralLocal:
    """Servi chez soi, Voxtral n'a ni facture ni envoi : l'interface ne doit pas les annoncer."""

    def test_avertissement_seulement_pour_l_api(self, monkeypatch):
        from voxlibris.web import app as app_module

        monkeypatch.setenv("VOXLIBRIS_MISTRAL_BASE_URL", "http://voxtral:8600/v1")
        assert app_module.voxtral_is_local()
        monkeypatch.setenv("VOXLIBRIS_MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
        assert not app_module.voxtral_is_local()


class TestPisteAJour:
    def test_la_piste_est_a_jour_par_son_contenu(self, tmp_path):
        import json

        from voxlibris.worker import track_is_current

        segments, track = tmp_path / "ch01.jsonl", tmp_path / "ch01.wav"
        segments.write_text('{"idx": 0, "text": "Un."}\n')
        assert not track_is_current(track, segments)

        track.write_bytes(b"RIFF")
        track.with_suffix(".timing.json").write_text(json.dumps([{"idx": 0, "text": "Un."}]))
        assert track_is_current(track, segments)

        # Le texte est corrigé et les segments repréparés : la piste ne le dit plus.
        segments.write_text('{"idx": 0, "text": "Un et deux."}\n')
        assert not track_is_current(track, segments)

    def test_la_page_de_synthese_donne_la_marche_a_suivre(self, client, make_epub):
        import os

        from voxlibris.web.app import project_dir

        name = TestRelecture._create(None, client, make_epub)
        root = project_dir(name)
        (root / "work" / "segments").mkdir(parents=True)
        segments = root / "work" / "segments" / "ch01.jsonl"
        segments.write_text(
            '{"idx": 0, "text": "Un.", "pause_after_ms": 0, "chapter": 1, "title": "Un"}\n'
        )
        assert (
            "Texte corrigé depuis la préparation" not in client.get(f"/projects/{name}/synth").text
        )

        later = segments.stat().st_mtime + 60
        text = root / "text" / "clean" / "ch01.md"
        os.utime(text, (later, later))
        page = client.get(f"/projects/{name}/synth").text
        assert "Texte corrigé depuis la préparation — chapitre 01" in page
        assert "texte corrigé" in page

        # Les segments encore à écouter restent accessibles malgré la correction.
        import json

        wav = root / "out" / "wav"
        wav.mkdir(parents=True)
        (wav / "ch01.wav").write_bytes(b"RIFF" + b"\0" * 60)
        (wav / "ch01.timing.json").write_text(
            json.dumps(
                [
                    {
                        "idx": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "clean": False,
                        "cause": "babil",
                        "attempts": 3,
                        "split": False,
                        "text": "Un.",
                    }
                ]
            ),
            encoding="utf-8",
        )
        page = client.get(f"/projects/{name}/synth").text
        assert "texte corrigé" in page and "1 à écouter" in page


class TestFichiers:
    """Les pistes s'écoutent dans la page ; les livres audio se téléchargent."""

    def test_une_piste_se_lit_en_place(self, client, make_epub):
        name = TestRelecture._create(None, client, make_epub)
        from voxlibris.web.app import project_dir

        wav = project_dir(name) / "out" / "wav"
        wav.mkdir(parents=True)
        (wav / "ch01.wav").write_bytes(b"RIFF" + b"\0" * 60)
        (project_dir(name) / "out" / "livre.m4b").write_bytes(b"m4b")
        response = client.get(f"/projects/{name}/files/wav/ch01.wav")
        assert response.status_code == 200
        assert response.headers["content-disposition"].startswith("inline")
        response = client.get(f"/projects/{name}/files/livre.m4b")
        assert response.headers["content-disposition"].startswith("attachment")
        # La page de synthèse propose la lecture et renvoie aux segments à écouter.
        response = client.get(f"/projects/{name}/synth")
        assert 'class="btn sm ghost listen"' in response.text
        assert 'id="player-bar"' in response.text


class TestValidation:
    def test_un_segment_valide_sort_de_la_liste(self, client, make_epub):
        import json

        from voxlibris.web.app import project_dir

        name = TestRelecture._create(None, client, make_epub)
        wav = project_dir(name) / "out" / "wav"
        wav.mkdir(parents=True)
        (wav / "ch01.wav").write_bytes(b"RIFF" + b"\0" * 60)
        entry = {
            "idx": 0,
            "start": 0.0,
            "end": 1.0,
            "clean": False,
            "cause": "babil",
            "attempts": 3,
            "split": False,
            "text": "Le gardien du phare.",
        }
        (wav / "ch01.timing.json").write_text(json.dumps([entry]), encoding="utf-8")

        page = client.get(f"/projects/{name}/synth").text
        assert "1 signalé" in page and "Valider" in page

        response = client.post(
            f"/projects/{name}/segments/approve",
            data={"chapter": 1, "idx": 0, "back": f"/projects/{name}/synth?chapter=1#flagged"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == f"/projects/{name}/synth?chapter=1#flagged"
        page = client.get(f"/projects/{name}/synth").text
        assert "0 signalé" in page and "1 validé" in page
        page = client.get(f"/projects/{name}/synth?cause=validé").text
        assert "Rétablir" in page and "babil" in page

        client.post(
            f"/projects/{name}/segments/approve", data={"chapter": 1, "idx": 0, "undo": "1"}
        )
        assert "1 signalé" in client.get(f"/projects/{name}/synth").text


class TestCouverture:
    def test_tiree_de_l_epub(self, client, make_epub):
        name = TestRelecture._create(None, client, make_epub)
        # Le livre de test n'a pas d'image de couverture.
        assert client.get(f"/projects/{name}/cover").status_code == 404

    def test_cherchee_en_ligne(self, client, make_epub, monkeypatch):
        name = TestRelecture._create(None, client, make_epub)
        monkeypatch.setattr(
            "voxlibris.ingest.metadata.search_cover", lambda title, author="", **k: b"J" * 5000
        )
        response = client.post(f"/projects/{name}/cover/search", follow_redirects=False)
        assert response.headers["location"].endswith("?cover=trouvee")
        assert client.get(f"/projects/{name}/cover").content == b"J" * 5000
        assert "Trouvée sur Open Library" in client.get(f"/projects/{name}?cover=trouvee").text

        monkeypatch.setattr("voxlibris.ingest.metadata.search_cover", lambda *a, **k: None)
        response = client.post(f"/projects/{name}/cover/search", follow_redirects=False)
        assert response.headers["location"].endswith("?cover=introuvable")


class TestClonage:
    """Un extrait déposé devient une voix : c'est un fichier dans le dossier des voix."""

    def test_depose_puis_retire(self, client, make_epub, tmp_path):
        from voxlibris.config import voices_dir

        name = TestRelecture._create(None, client, make_epub)
        response = client.post(
            f"/projects/{name}/voices/clone",
            data={"label": "Marie Dupont"},
            files={"file": ("enregistrement.wav", b"RIFF" + b"\0" * 100, "audio/wav")},
            follow_redirects=False,
        )
        assert response.status_code == 303 and "cloning=deposee" in response.headers["location"]
        saved = voices_dir() / "marie-dupont.wav"
        assert saved.read_bytes().startswith(b"RIFF")

        page = client.get(f"/projects/{name}/voices?cloning=deposee").text
        assert "marie dupont" in page and "extrait déposé" in page

        # Un autre format sous le même nom remplace l'extrait, sans doublon.
        client.post(
            f"/projects/{name}/voices/clone",
            data={"label": "Marie Dupont"},
            files={"file": ("autre.mp3", b"ID3" + b"\0" * 50, "audio/mpeg")},
        )
        assert not saved.exists() and (voices_dir() / "marie-dupont.mp3").exists()

        client.post(f"/projects/{name}/voices/clone/delete", data={"sample": "marie-dupont.mp3"})
        assert not (voices_dir() / "marie-dupont.mp3").exists()

    def test_rien_n_est_coche_d_office_une_fois_le_banc_entame(self, client, make_epub):
        from voxlibris.web import app as app_module

        name = TestRelecture._create(None, client, make_epub)
        project = app_module.load_project(name)
        checked = [v for _, v, _, on in app_module.sample_candidates("fr", project) if on]
        assert checked == ["Viktor Menelaos", "Damien Black"]

        (project.out_dir / "samples").mkdir(parents=True)
        (project.out_dir / "samples" / "xtts--Viktor_Menelaos.wav").write_bytes(b"RIFF")
        assert not [on for *_, on in app_module.sample_candidates("fr", project) if on]

    def test_le_nom_vient_du_fichier_a_defaut(self, client, make_epub):
        from voxlibris.config import voices_dir

        name = TestRelecture._create(None, client, make_epub)
        client.post(
            f"/projects/{name}/voices/clone",
            files={"file": ("Voix_Grave.flac", b"fLaC", "audio/flac")},
        )
        assert (voices_dir() / "voix-grave.flac").exists()

    def test_refuse_ce_qui_n_est_pas_du_son(self, client, make_epub):
        name = TestRelecture._create(None, client, make_epub)
        response = client.post(
            f"/projects/{name}/voices/clone",
            files={"file": ("voix.txt", b"bonjour", "text/plain")},
        )
        assert response.status_code == 400

    def test_les_voix_deposees_passent_devant_les_voix_d_exemple(self, monkeypatch):
        from voxlibris.web import app as app_module

        monkeypatch.setenv("VOXLIBRIS_ZONOS2_BASE_URL", "http://zonos2:1919")
        monkeypatch.setattr(
            "voxlibris.tts.zonos2.voice_names",
            lambda *a, **k: ["AmericanFemale", "AmericanMale", "BritishFemale", "marie", "Paul"],
        )
        assert app_module.zonos2_voices() == [
            "marie",
            "Paul",
            "AmericanFemale",
            "AmericanMale",
            "BritishFemale",
        ]

    def test_le_moteur_est_presente_avec_ses_voix(self, client, make_epub, monkeypatch):
        from voxlibris.web import app as app_module

        name = TestRelecture._create(None, client, make_epub)
        monkeypatch.setattr(app_module, "zonos2_voices", lambda: ["Marie", "American Female"])
        page = client.get(f"/projects/{name}/voices").text
        assert "ZONOS2" in page and "clonage de voix" in page
        assert "ZONOS2 · Marie" in page

    def test_la_sonde_des_reglages(self, client, monkeypatch):
        from test_zonos2 import CAPABILITIES, SPEAKERS, FakeClient

        from voxlibris.tts import zonos2

        real = zonos2.Client
        fake = FakeClient(
            {"/tts/capabilities": (CAPABILITIES, {}), "/tts/speakers": (SPEAKERS, {})}
        )
        monkeypatch.setattr(zonos2, "Client", lambda *a, **k: fake)
        assert "2 voix" in client.post("/settings/test/zonos2").text

        # Sans adresse, le vrai client refuse de partir, et la page dit quoi renseigner.
        monkeypatch.setattr(zonos2, "Client", real)
        monkeypatch.setenv("VOXLIBRIS_ZONOS2_BASE_URL", "")
        assert "VOXLIBRIS_ZONOS2_BASE_URL" in client.post("/settings/test/zonos2").text

    def test_omnivoice_est_presente_avec_ses_voix(self, client, make_epub, monkeypatch):
        from voxlibris.web import app as app_module

        name = TestRelecture._create(None, client, make_epub)
        monkeypatch.setattr(app_module, "zonos2_voices", lambda: [])
        monkeypatch.setattr(app_module, "omnivoice_voices", lambda: ["Marie"])
        page = client.get(f"/projects/{name}/voices").text
        assert "OmniVoice · Marie" in page and "OmniVoice" in page

    def test_la_sonde_omnivoice(self, client, monkeypatch):
        from test_omnivoice import HEALTH, VOICES, FakeClient

        from voxlibris.tts import omnivoice

        real = omnivoice.Client
        fake = FakeClient({"/health": (HEALTH, {}), "/voices": (VOICES, {})})
        monkeypatch.setattr(omnivoice, "Client", lambda *a, **k: fake)
        assert "cuda · 2 voix" in client.post("/settings/test/omnivoice").text

        monkeypatch.setattr(omnivoice, "Client", real)
        monkeypatch.setenv("VOXLIBRIS_OMNIVOICE_BASE_URL", "")
        assert "VOXLIBRIS_OMNIVOICE_BASE_URL" in client.post("/settings/test/omnivoice").text

    def test_un_serveur_absent_n_est_cherche_qu_une_fois_par_visite(self, monkeypatch):
        from voxlibris.web import app as app_module

        calls = []
        monkeypatch.setattr(app_module, "zonos2_voices", lambda: calls.append(1) or [])
        monkeypatch.setattr(app_module, "omnivoice_voices", lambda: ["Marie"])
        monkeypatch.setattr(app_module, "voxtral_voices", lambda language="": [])
        first = app_module.catalogues("fr")
        second = app_module.catalogues("fr")
        assert first["omnivoice"] == ["Marie"] and second == first
        assert calls == [1]

        # Un dépôt de voix vide la mémoire : la page suivante redemande.
        app_module.forget_catalogues()
        app_module.catalogues("fr")
        assert calls == [1, 1]

    def test_la_page_voix_ne_demande_les_catalogues_qu_une_fois(
        self, client, make_epub, monkeypatch
    ):
        from voxlibris.web import app as app_module

        name = TestRelecture._create(None, client, make_epub)
        calls = []
        monkeypatch.setattr(app_module, "zonos2_voices", lambda: calls.append(1) or ["Marie"])
        monkeypatch.setattr(app_module, "omnivoice_voices", lambda: [])
        monkeypatch.setattr(app_module, "voxtral_voices", lambda language="": [])
        assert "ZONOS2 · Marie" in client.get(f"/projects/{name}/voices").text
        assert calls == [1]


class TestConcierge:
    """Ce que l'interface montre des serveurs que l'atelier réveille et endort."""

    def _pulse(self, monkeypatch, served, engines=None):
        from voxlibris import atelier

        pulse = {
            "online": True,
            "age": 1,
            "busy": None,
            "device": "cuda",
            "gpu": "RTX",
            "docker": True,
            "served": served,
            "idle_minutes": 15,
            "engines": engines
            or {
                "xtts": True,
                "kokoro": True,
                "piper": True,
                "voxtral": False,
                "zonos2": True,
                "omnivoice": False,
            },
        }
        monkeypatch.setattr(atelier, "status", lambda root: pulse)

    def test_les_moteurs_absents_ne_sont_pas_proposes(self, client, make_epub, monkeypatch):
        from voxlibris.web import app as app_module

        name = TestRelecture._create(None, client, make_epub)
        self._pulse(monkeypatch, {"zonos2": {"state": "stopped", "name": "voxlibris-zonos2-1"}})
        monkeypatch.setattr(
            app_module, "served_state", lambda e: "stopped" if e == "zonos2" else ""
        )
        monkeypatch.setattr(app_module, "voxtral_voices", lambda language="": [])
        monkeypatch.setattr(app_module, "omnivoice_voices", lambda: [])
        (app_module.voices_dir()).mkdir(parents=True)
        (app_module.voices_dir() / "marie.wav").write_bytes(b"RIFF")
        page = client.get(f"/projects/{name}/voices").text
        assert "en veille" in page and "ZONOS2 · marie" in page
        # Ni tuile, ni ligne au banc, ni option de saisie pour les moteurs que l'atelier n'a pas.
        assert "Voxtral TTS" not in page and 'value="omnivoice"' not in page
        assert "OmniVoice · " not in page and "XTTS-v2" in page

    def test_liberer_la_carte_depose_une_tache(self, client, monkeypatch):
        from voxlibris.web import app as app_module

        self._pulse(monkeypatch, {"zonos2": {"state": "running", "name": "voxlibris-zonos2-1"}})
        assert "Libérer la carte" in client.get("/settings").text
        response = client.post("/atelier/release", follow_redirects=False)
        assert response.status_code == 303
        job = app_module.queue.active("atelier")
        assert job is not None and job.kind == "release"
        # Une seconde demande n'en dépose pas une autre.
        client.post("/atelier/release", follow_redirects=False)
        assert len([j for j in app_module.queue.list("atelier") if j.kind == "release"]) == 1
        assert "atelier" in client.get("/jobs").text


class TestPagesDOrigine:
    """Un EPUB paginé montre ses pages en regard du texte, dans un cadre isolé."""

    @pytest.fixture
    def name(self, client, tmp_path):
        from test_ingest_epub_fixe import page, write_epub

        first = page([("x0", 600, "h0", "<span>Première page du registre, assez longue.</span>")])
        second = page([("x0", 600, "h0", "<span>Seconde page du registre, assez longue.</span>")])
        path = write_epub(
            tmp_path / "registre.epub",
            [first, second],
            [(1, "1. Le début")],
            {"fonts/a.ttf": b"\x00\x01\x00\x00police", "pdf2fl.js": b"alert(1)"},
        )
        with path.open("rb") as handle:
            client.post("/projects", files={"file": (path.name, handle)})
        return "le-registre"

    def test_la_relecture_ouvre_un_cadre(self, client, name):
        response = client.get(f"/projects/{name}/review/1")
        assert response.status_code == 200
        assert '<iframe id="page-image"' in response.text
        assert 'sandbox="allow-same-origin"' in response.text
        assert 'data-width="1000"' in response.text

    def test_la_page_et_ses_ressources(self, client, name):
        response = client.get(f"/projects/{name}/page/1?page=1")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert f'<base href="/projects/{name}/source/OPS/">' in response.text
        assert "<script" not in response.text
        assert "script" not in response.headers["content-security-policy"].replace("'none'", "")
        assert "default-src 'none'" in response.headers["content-security-policy"]
        assert client.get(f"/projects/{name}/source/OPS/fonts/a.ttf").status_code == 200
        assert client.get(f"/projects/{name}/source/OPS/pdf2fl.js").status_code == 404
        assert client.get(f"/projects/{name}/source/OPS/content.opf").status_code == 404

    def test_hors_des_bornes_on_reste_dans_le_chapitre(self, client, name):
        response = client.get(f"/projects/{name}/page/1?page=99")
        assert response.status_code == 200
        assert "Seconde page" in response.text
