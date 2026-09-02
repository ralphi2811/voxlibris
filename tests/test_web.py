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
        """La prise en charge fait office de verrou entre ouvriers concurrents."""
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
        assert "Aucun projet" in response.text

    def test_depot_dun_epub(self, client, make_epub):
        path = make_epub(["Le départ", "La traversée"])
        with path.open("rb") as handle:
            response = client.post("/projects", files={"file": (path.name, handle)})
        assert response.status_code == 200  # redirection suivie
        assert "Le registre du gardien" in response.text
        # Un EPUB n'a rien à relire.
        assert "relecture nécessaire" not in response.text

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
        assert "Aucun projet" in response.text
        assert client.get(f"/projects/{name}").status_code == 404

    def test_refuse_pendant_une_tache(self, client, make_epub):
        """Supprimer sous les pieds de l'ouvrier le ferait écrire dans le vide."""
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

    def test_chapitre_inconnu(self, client, make_epub):
        name = self._create(client, make_epub)
        assert client.post(f"/projects/{name}/chapters/9/delete").status_code == 404

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
        second = client.post(
            f"/projects/{name}/jobs/normalize", follow_redirects=False
        )
        assert second.status_code == 409
