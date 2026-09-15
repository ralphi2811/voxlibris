"""Tests de la lecture des scans sans couche de texte : le client du service RapidOCR,
la couche de texte posée sur le PDF, et la tâche de l'atelier qui les enchaîne.

Le serveur lui-même n'est jamais appelé : un faux client rend des lignes connues, et
c'est ce qu'on en fait — position, redressement, extraction — que l'on vérifie.
"""

from __future__ import annotations

import io
import json
import math
import urllib.error
from pathlib import Path

import pymupdf
import pytest

from voxlibris import ingest
from voxlibris.ingest import Kind, ocr
from voxlibris.ingest.ocr import Client, Line, OCRError, layer, ocr_pdf, skew

DPI = 200
PX = DPI / 72.0


def box(x: float, y: float, width: float, height: float, angle: float = 0.0) -> tuple:
    """Le quadrilatère, en pixels, d'une ligne posée en points, tournée d'un angle."""
    cos, sin = math.cos(angle), math.sin(angle)
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    return tuple(
        ((x + cx * cos - cy * sin) * PX, (y + cx * sin + cy * cos) * PX) for cx, cy in corners
    )


def line(text: str, x: float, y: float, width: float, height: float = 12.0, **kw) -> Line:
    return Line(text, box(x, y, width, height, kw.pop("angle", 0.0)), kw.pop("score", 1.0))


class FakeClient:
    """Rend, page après page, les lignes qu'on lui a confiées."""

    def __init__(self, pages: list[list[Line]]) -> None:
        self.pages = pages
        self.seen: list[int] = []

    def probe(self) -> dict:
        return {"ready": True, "lang": "latin"}

    def recognize(self, image: bytes) -> list[Line]:
        self.seen.append(len(image))
        return self.pages[len(self.seen) - 1] if len(self.seen) <= len(self.pages) else []


def blank_pdf(path: Path, pages: int = 1, width: float = 400, height: float = 600) -> Path:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page(width=width, height=height)
    doc.save(str(path))
    return path


def spans(page: pymupdf.Page) -> list[dict]:
    return [
        span
        for block in page.get_text("dict")["blocks"]
        if block["type"] == 0
        for found in block["lines"]
        for span in found["spans"]
    ]


class TestGeometrie:
    def test_hauteur_largeur_et_angle_dune_ligne(self):
        found = line("texte", 10, 20, 100, 10, angle=math.radians(3))
        assert found.width == pytest.approx(100 * PX, rel=1e-3)
        assert found.height == pytest.approx(10 * PX, rel=1e-3)
        assert found.angle == pytest.approx(math.radians(3), abs=1e-6)

    def test_inclinaison_lue_sur_la_marge(self):
        # Une page tournée de deux degrés : la marge gauche recule en descendant.
        tilt = math.radians(2)
        lines = [
            line("x" * 40, 40 - y * math.tan(tilt), y, 300, angle=tilt) for y in range(60, 500, 20)
        ]
        assert skew(lines) == pytest.approx(tilt, abs=math.radians(0.1))

    def test_sans_longue_ligne_pas_dinclinaison(self):
        assert skew([line("a", 10, 10, 10)]) == 0.0
        assert skew([]) == 0.0


class TestCouche:
    def test_le_texte_pose_se_relit_a_sa_place(self, tmp_path):
        doc = pymupdf.open()
        page = doc.new_page(width=400, height=600)
        placed = layer(
            page,
            [
                line("Première ligne du récit, qui court jusqu'au bord droit.", 40, 60, 300),
                line("Deuxième ligne, plus bas, tout aussi longue que l'autre.", 40, 80, 300),
                line("bruit", 200, 300, 40, score=0.1),
            ],
            DPI,
            ocr.font_file(),
        )
        assert placed == 2
        found = spans(page)
        assert [s["text"] for s in found] == [
            "Première ligne du récit, qui court jusqu'au bord droit.",
            "Deuxième ligne, plus bas, tout aussi longue que l'autre.",
        ]
        # Invisible, mais bien là : la boîte du texte recouvre celle de la ligne lue.
        assert found[0]["bbox"][1] == pytest.approx(60, abs=6)
        assert found[1]["bbox"][1] - found[0]["bbox"][1] == pytest.approx(20, abs=3)
        assert found[0]["bbox"][2] - found[0]["bbox"][0] == pytest.approx(300, abs=12)
        assert 6 < found[0]["size"] < 12

    def test_les_caracteres_typographiques_survivent(self):
        doc = pymupdf.open()
        page = doc.new_page(width=400, height=600)
        text = "Bœuf d’été — « ça » “oui” …"
        layer(page, [line(text, 40, 60, 300)], DPI, ocr.font_file())
        (found,) = spans(page)
        if ocr.font_file() is None:
            assert found["text"] == 'Boeuf d\'été - « ça » "oui" ...'
        else:
            assert found["text"] == text

    def test_une_page_de_travers_est_redressee(self):
        doc = pymupdf.open()
        page = doc.new_page(width=400, height=600)
        tilt = math.radians(2.5)
        lines = [
            line(f"ligne {i}", 40 - y * math.tan(tilt), y, 300, angle=tilt)
            for i, y in enumerate(range(60, 500, 20))
        ]
        layer(page, lines, DPI, None)
        lefts = [s["bbox"][0] for s in spans(page)]
        assert max(lefts) - min(lefts) < 3.0

    def test_le_pdf_produit_passe_pour_un_scan_ocerise(self, tmp_path):
        source = blank_pdf(tmp_path / "scan.pdf", pages=2)
        assert ingest.detect_kind(source) is Kind.PDF_IMAGE
        target = tmp_path / "scan.ocr.pdf"
        client = FakeClient([[line("Une ligne.", 40, 60, 300)], []])
        stats = ocr_pdf(source, target, client)
        assert stats["pages"] == 2 and stats["lines"] == 1 and len(client.seen) == 2
        assert ingest.detect_kind(target) is Kind.PDF_OCR
        assert pymupdf.open(target).metadata["producer"].startswith(ingest.OCR_PRODUCER)


class TestBoutEnBout:
    def test_du_scan_au_document(self, tmp_path):
        source = blank_pdf(tmp_path / "scan.pdf", pages=2)
        page1 = [
            line("Il était une fois, dans un pays lointain, un roi", 40, 60, 320),
            line("qui n'avait pas d'enfant. Il en rêvait chaque nuit.", 40, 80, 320),
            line("Un matin, la reine vint le trouver dans la salle du", 40, 100, 320),
            line("trône, et lui dit quelque chose de merveilleux.", 40, 120, 320),
            line("— Sire, nous aurons un enfant à la fin de l'hiver.", 40, 140, 240),
            line("Le roi n'en crut pas ses oreilles.", 40, 160, 220),
            line("Il fit sonner toutes les cloches du royaume, et le", 40, 180, 320),
            line("peuple accourut sur la place pour savoir pourquoi.", 40, 200, 320),
        ]
        page2 = [
            line("La nouvelle se répandit en un jour dans les campa-", 40, 60, 320),
            line("gnes, et chacun voulut voir la reine.", 40, 80, 250),
            line("Le printemps vint enfin, et l'enfant avec lui. On", 40, 100, 320),
            line("le prénomma Aubin, du nom de son grand-père, et", 40, 120, 320),
            line("toute la cour se pressa pour le voir dans son", 40, 140, 320),
            line("berceau de bois doré, sous les grandes fenêtres.", 40, 160, 320),
        ]
        target = tmp_path / "scan.ocr.pdf"
        ocr_pdf(source, target, FakeClient([page1, page2]))
        document = ingest.ingest(target)
        assert document.needs_review and document.notes["kind"] == "pdf-scan-ocr"
        (chapter,) = document.chapters
        assert chapter.paragraphs == [
            "Il était une fois, dans un pays lointain, un roi qui n'avait pas d'enfant. "
            "Il en rêvait chaque nuit. Un matin, la reine vint le trouver dans la salle du "
            "trône, et lui dit quelque chose de merveilleux.",
            "— Sire, nous aurons un enfant à la fin de l'hiver.",
            "Le roi n'en crut pas ses oreilles.",
            "Il fit sonner toutes les cloches du royaume, et le peuple accourut sur la "
            "place pour savoir pourquoi. La nouvelle se répandit en un jour dans les "
            "campagnes, et chacun voulut voir la reine.",
            "Le printemps vint enfin, et l'enfant avec lui. On le prénomma Aubin, du nom "
            "de son grand-père, et toute la cour se pressa pour le voir dans son berceau "
            "de bois doré, sous les grandes fenêtres.",
        ]


class TestClient:
    def test_sans_adresse_le_client_le_dit(self, monkeypatch):
        monkeypatch.setattr(ocr, "base_url", lambda env=None: "")
        with pytest.raises(OCRError, match="profile rapidocr"):
            Client()

    def test_les_lignes_du_serveur_sont_relues(self, monkeypatch):
        sent: dict = {}

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def fake_urlopen(request, timeout=0):
            sent["url"] = request.full_url
            sent["type"] = request.get_header("Content-type")
            sent["body"] = request.data
            payload = {
                "lines": [
                    {"text": "Bonjour", "box": [[0, 0], [50, 0], [50, 10], [0, 10]], "score": 0.9}
                ]
            }
            return Response(json.dumps(payload).encode())

        monkeypatch.setattr(ocr.urllib.request, "urlopen", fake_urlopen)
        lines = Client("http://ocr:1921/").recognize(b"PNG")
        assert sent == {"url": "http://ocr:1921/ocr", "type": "image/png", "body": b"PNG"}
        assert lines == [Line("Bonjour", ((0, 0), (50, 0), (50, 10), (0, 10)), 0.9)]

    def test_un_refus_du_serveur_devient_une_erreur_lisible(self, monkeypatch):
        def fake_urlopen(request, timeout=0):
            raise urllib.error.HTTPError(
                request.full_url, 413, "trop", {}, io.BytesIO(b"Image trop lourde.")
            )

        monkeypatch.setattr(ocr.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(OCRError, match="413.*trop lourde"):
            Client("http://ocr:1921").recognize(b"PNG")

    def test_un_serveur_absent_devient_une_erreur_lisible(self, monkeypatch):
        def fake_urlopen(request, timeout=0):
            raise urllib.error.URLError("connexion refusée")

        monkeypatch.setattr(ocr.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(OCRError, match="injoignable"):
            Client("http://ocr:1921").probe()


class TestTache:
    """La tâche de l'atelier : du projet vide au texte brut à relire."""

    def _project(self, tmp_path):
        from voxlibris.document import Document
        from voxlibris.project import Project

        source = (
            blank_pdf(tmp_path / "depot" / "livre.pdf", pages=1)
            if (tmp_path / "depot").mkdir() is None
            else None
        )
        document = Document(
            title="Le livre",
            author="A.",
            source=source,
            needs_review=True,
            notes={"kind": "pdf-scan-image"},
        )
        return Project.create(tmp_path / "p", document)

    def test_le_scan_lu_devient_le_texte_brut(self, tmp_path, monkeypatch):
        from voxlibris import worker
        from voxlibris.web.jobs import Queue

        project = self._project(tmp_path)
        assert not list(project.raw_dir.glob("ch*.md"))
        page = [
            line("Une première ligne assez longue pour compter.", 40, 60, 320),
            line("Et une seconde, qui la suit sans alinéa.", 40, 80, 320),
        ]
        monkeypatch.setattr(ocr, "Client", lambda *a, **k: FakeClient([page]))
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("p", "ocr")
        worker.run_ocr(project, job, queue)

        text = (project.raw_dir / "ch01.md").read_text(encoding="utf-8")
        assert "Une première ligne assez longue pour compter. Et une seconde" in text
        assert project.source.endswith("livre.ocr.pdf") and Path(project.source).exists()
        assert project.needs_review and project.notes["OCR"].startswith("RapidOCR : 2 lignes")
        assert "chapitre(s), à relire" in queue.get(job.id).log

    def test_un_texte_relu_nest_pas_ecrase_sans_ordre(self, tmp_path, monkeypatch):
        from voxlibris import worker
        from voxlibris.web.jobs import Queue

        project = self._project(tmp_path)
        project.clean_dir.mkdir(parents=True)
        (project.clean_dir / "ch01.md").write_text(
            '---\nchapter: 1\ntitle: "Relu"\n---\n\nRelu.\n', encoding="utf-8"
        )
        monkeypatch.setattr(
            ocr, "Client", lambda *a, **k: FakeClient([[line("Nouveau.", 40, 60, 300)]])
        )
        queue = Queue(tmp_path / "jobs.sqlite")
        with pytest.raises(RuntimeError, match="déjà été relu"):
            worker.run_ocr(project, queue.enqueue("p", "ocr"), queue)
        worker.run_ocr(project, queue.enqueue("p", "ocr", force=True), queue)
        assert (project.clean_dir / "ch01.md").read_text(encoding="utf-8").endswith("Relu.\n")
        assert "Nouveau." in (project.raw_dir / "ch01.md").read_text(encoding="utf-8")


class TestVoixInventee:
    """La tâche qui invente une voix chez OmniVoice et la dépose comme extrait."""

    def test_lextrait_rejoint_le_dossier_des_voix(self, tmp_path, monkeypatch):
        import wave

        import numpy as np

        from voxlibris import worker
        from voxlibris.config import voices_dir
        from voxlibris.tts import omnivoice
        from voxlibris.web.jobs import Queue

        asked: dict = {}

        class FakeOmni:
            def __init__(self, *a, **k):
                pass

            def probe(self):
                return {"ready": True}

            def design(self, text, instruct, language="fr", speed=1.0):
                asked.update(text=text, instruct=instruct, language=language)
                return np.full(2400, 0.5, dtype=np.float32), 24000

        monkeypatch.setattr(omnivoice, "Client", FakeOmni)
        project = TestTache()._project(tmp_path)
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("p", "design", label="vieille-dame", instruct="female, elderly")
        worker.run_design(project, job, queue)

        target = voices_dir() / "vieille-dame.wav"
        assert target.exists()
        with wave.open(str(target)) as saved:
            assert saved.getframerate() == 24000 and saved.getnframes() == 2400
        assert asked == {
            "text": omnivoice.design_text("fr"),
            "instruct": "female, elderly",
            "language": "fr",
        }
        assert "0.1 s déposées" in queue.get(job.id).log
        assert "pas encore de segments" in queue.get(job.id).log

    def test_avec_des_segments_la_voix_passe_au_banc(self, tmp_path, monkeypatch):
        import numpy as np
        from test_atelier import make_project

        from voxlibris import worker
        from voxlibris.normalize import build_segments
        from voxlibris.tts import omnivoice
        from voxlibris.web.jobs import Queue

        class FakeOmni:
            def __init__(self, *a, **k):
                pass

            def probe(self):
                return {"ready": True}

            def design(self, text, instruct, language="fr", speed=1.0):
                return np.full(2400, 0.5, dtype=np.float32), 24000

        class FakeEngine:
            name, voice, speed, sample_rate = "omnivoice", "grand mere", 1.0, 24000
            supports_speed = True
            loaded: list = []

            def say(self, text):
                return np.full(len(text) * 2400, 0.3, np.float32)

        def fake_load(backend, voice, device=None):
            FakeEngine.loaded.append((backend, voice))
            return FakeEngine()

        monkeypatch.setattr(omnivoice, "Client", FakeOmni)
        monkeypatch.setattr("voxlibris.tts.backends.load", fake_load)
        project = make_project(tmp_path / "p")
        build_segments(project.text_dir, project.segments_dir)
        queue = Queue(tmp_path / "jobs.sqlite")
        job = queue.enqueue("p", "design", label="grand-mere", instruct="female, elderly")
        worker.run_design(project, job, queue)

        assert FakeEngine.loaded == [("omnivoice", "grand mere")]
        assert (project.out_dir / "samples" / "omnivoice--grand_mere.wav").exists()
        assert "échantillons prêts" in queue.get(job.id).log

    def test_sans_nom_ni_trait_la_tache_refuse(self, tmp_path):
        from voxlibris import worker
        from voxlibris.web.jobs import Queue

        project = TestTache()._project(tmp_path)
        queue = Queue(tmp_path / "jobs.sqlite")
        with pytest.raises(RuntimeError, match="nom"):
            worker.run_design(project, queue.enqueue("p", "design", label="x"), queue)
