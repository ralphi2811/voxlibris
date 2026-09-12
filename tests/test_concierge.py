"""Le concierge, éprouvé sans Docker ni carte.

Un faux relais tient l'inventaire et note ce qu'on lui demande ; la sonde des serveurs
est remplacée. Ce qui se teste est la logique : qui réveiller, qui endormir pour faire
de la place, quand rendre la carte, et ce que le battement publie.
"""

from __future__ import annotations

import json

import pytest

from voxlibris import concierge
from voxlibris.concierge import Concierge, DockerError


class FakeDocker:
    def __init__(self, containers: dict[str, str]):
        """containers : service → « running » ou « exited »."""
        self.states = dict(containers)
        self.calls: list[tuple[str, str]] = []

    def containers(self, project=None):
        return [
            {
                "Names": [f"/voxlibris-{service}-1"],
                "State": state,
                "Labels": {concierge.LABEL_SERVICE: service, concierge.LABEL_PROJECT: "voxlibris"},
            }
            for service, state in self.states.items()
        ]

    def start(self, ident):
        self.calls.append(("start", ident))
        self.states[ident.split("-")[1]] = "running"

    def stop(self, ident, grace_s=30):
        self.calls.append(("stop", ident))
        self.states[ident.split("-")[1]] = "exited"


@pytest.fixture
def awake(monkeypatch):
    """Les serveurs répondent dès qu'ils sont démarrés."""
    monkeypatch.setattr(concierge, "probe", lambda served, timeout=5.0: True)
    monkeypatch.setattr(concierge.time, "sleep", lambda s: None)


class TestInventaire:
    def test_etat_de_chaque_moteur(self):
        docker = FakeDocker({"zonos2": "exited", "omnivoice": "running"})
        c = Concierge(docker, "voxlibris", vram_gb=24)
        assert c.inventory() == {
            "voxtral": {"state": "absent", "name": ""},
            "zonos2": {"state": "stopped", "name": "voxlibris-zonos2-1"},
            "omnivoice": {"state": "running", "name": "voxlibris-omnivoice-1"},
        }
        assert c.running() == ["omnivoice"]

    def test_sans_relais_rien(self):
        c = Concierge(None)
        assert not c.active and c.inventory() == {} and c.release() == []
        c.ensure("zonos2")  # ne lève pas


class TestReveil:
    def test_reveille_un_serveur_endormi(self, awake):
        docker = FakeDocker({"omnivoice": "exited"})
        c = Concierge(docker, "voxlibris", vram_gb=24)
        said = []
        c.ensure("omnivoice", said.append)
        assert docker.calls == [("start", "voxlibris-omnivoice-1")]
        assert any("réveil de omnivoice" in s for s in said) and "prêt" in said[-1]

    def test_ne_touche_pas_a_un_serveur_qui_tourne(self, awake):
        docker = FakeDocker({"omnivoice": "running"})
        c = Concierge(docker, "voxlibris", vram_gb=24)
        c.ensure("omnivoice")
        assert docker.calls == []

    def test_endort_ce_qui_ne_tient_pas_a_cote(self, awake):
        # ZONOS2 (21 Go) tourne ; Voxtral (16 Go) ne tient pas à côté sur 24 Go.
        docker = FakeDocker({"zonos2": "running", "voxtral": "exited", "omnivoice": "running"})
        c = Concierge(docker, "voxlibris", vram_gb=24)
        said = []
        c.ensure("voxtral", said.append)
        assert docker.calls == [("stop", "voxlibris-zonos2-1"), ("start", "voxlibris-voxtral-1")]
        # OmniVoice (3 Go) tient à côté de Voxtral : il reste.
        assert docker.states["omnivoice"] == "running"
        assert any("mise en veille de zonos2" in s for s in said)

    def test_un_moteur_embarque_fait_de_la_place_aussi(self, awake):
        docker = FakeDocker({"zonos2": "running"})
        c = Concierge(docker, "voxlibris", vram_gb=24)
        c.ensure("xtts")
        assert docker.calls == [("stop", "voxlibris-zonos2-1")]

    def test_un_moteur_sans_conteneur_est_laisse(self, awake):
        docker = FakeDocker({"zonos2": "running"})
        c = Concierge(docker, "voxlibris", vram_gb=24)
        c.ensure("voxtral")  # API distante, ou serveur externe : rien à réveiller
        assert docker.calls == []

    def test_un_reveil_qui_n_aboutit_pas_le_dit(self, monkeypatch):
        monkeypatch.setattr(concierge, "probe", lambda served, timeout=5.0: False)
        monkeypatch.setattr(concierge, "WAKE_TIMEOUT", 0.0)
        monkeypatch.setattr(concierge.time, "sleep", lambda s: None)
        c = Concierge(FakeDocker({"zonos2": "exited"}), "voxlibris", vram_gb=24)
        with pytest.raises(DockerError, match="docker logs voxlibris-zonos2-1"):
            c.ensure("zonos2")


class TestSommeil:
    def test_rend_la_carte(self):
        docker = FakeDocker({"zonos2": "running", "omnivoice": "running", "voxtral": "exited"})
        c = Concierge(docker, "voxlibris", vram_gb=24)
        assert sorted(c.release()) == ["omnivoice", "zonos2"]
        assert sorted(docker.calls) == [
            ("stop", "voxlibris-omnivoice-1"),
            ("stop", "voxlibris-zonos2-1"),
        ]

    def test_apres_le_delai_seulement(self, monkeypatch):
        monkeypatch.setenv("VOXLIBRIS_GPU_IDLE_MIN", "15")
        docker = FakeDocker({"zonos2": "running"})
        c = Concierge(docker, "voxlibris", vram_gb=24)
        assert c.maybe_release(busy=False) == []
        c.last_activity -= 16 * 60
        assert c.maybe_release(busy=True) == []  # une tâche tourne : on ne touche à rien
        assert c.maybe_release(busy=False) == ["zonos2"]

    def test_zero_veut_dire_jamais(self, monkeypatch):
        monkeypatch.setenv("VOXLIBRIS_GPU_IDLE_MIN", "0")
        c = Concierge(FakeDocker({"zonos2": "running"}), "voxlibris", vram_gb=24)
        c.last_activity -= 10 * 3600
        assert c.maybe_release(busy=False) == []


class TestBattement:
    def test_ce_que_le_battement_publie(self, monkeypatch):
        monkeypatch.setenv("VOXLIBRIS_GPU_IDLE_MIN", "15")
        c = Concierge(FakeDocker({"zonos2": "running"}), "voxlibris", vram_gb=24)
        seen = c.snapshot()
        assert seen["docker"] is True and seen["served"]["zonos2"]["state"] == "running"
        assert 0 < seen["release_in_s"] <= 15 * 60

    def test_l_adresse_decoule_du_conteneur(self, monkeypatch, tmp_path):
        from voxlibris import atelier
        from voxlibris.tts import omnivoice, zonos2

        monkeypatch.setenv("VOXLIBRIS_DATA", str(tmp_path))
        monkeypatch.setenv("VOXLIBRIS_ZONOS2_BASE_URL", "")
        monkeypatch.setenv("VOXLIBRIS_OMNIVOICE_BASE_URL", "")
        assert zonos2.base_url() == "" and concierge.implied_url("zonos2") == ""

        monkeypatch.setattr(atelier, "hardware", lambda: {"device": "cpu", "gpu": ""})
        c = Concierge(FakeDocker({"zonos2": "exited"}), "voxlibris", vram_gb=24)
        atelier.beat(tmp_path, None, {"engines": {}}, c)
        seen = json.loads(atelier.heartbeat_file(tmp_path).read_text())
        assert seen["served"]["zonos2"]["state"] == "stopped"
        # Un conteneur, même endormi, suffit : l'adresse va de soi, le moteur est offert.
        assert zonos2.base_url() == "http://zonos2:1919"
        assert concierge.served_state("zonos2") == "stopped"
        assert omnivoice.base_url() == ""
        assert seen["engines"]["zonos2"] is True and seen["engines"]["omnivoice"] is False

        # Le réglage explicite garde la main sur l'adresse déduite.
        monkeypatch.setenv("VOXLIBRIS_ZONOS2_BASE_URL", "http://ailleurs:1919")
        assert zonos2.base_url() == "http://ailleurs:1919"
