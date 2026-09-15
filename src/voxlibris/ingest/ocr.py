"""Lecture des PDF scannés sans couche de texte, par le service RapidOCR.

Un scan « image pure » n'a rien à extraire : chaque page n'est qu'une photographie. Le
service RapidOCR (profil « rapidocr » du docker-compose.yml, serveur
`docker/rapidocr-server.py`) y lit les lignes, avec leur position en pixels. Plutôt que
d'en faire un texte à plat, ce module **pose ces lignes sur le PDF lui-même**, en texte
invisible calé sur l'image — exactement la couche que portent les scans d'Internet
Archive. Le fichier obtenu prend ensuite le chemin ordinaire du scan océrisé,
`pdf_scan.py` : mesure de la mise en page, en-têtes courants, chapitrage, images des
pages en regard du texte à la relecture. Rien de tout cela n'a été récrit.

Deux précautions font la différence sur un vrai scan :

- **L'inclinaison.** Une page scannée de travers décale la marge gauche d'une ligne à
  l'autre, et `pdf_scan` prendrait chaque ligne pour un alinéa. L'angle médian des lignes
  est mesuré, et les positions redressées avant d'être écrites.
- **La police.** Le texte invisible doit rester lisible par PyMuPDF : les polices de base
  du PDF ignorent « œ », les guillemets typographiques et le tiret cadratin des dialogues.
  Une police TrueType de la machine est embarquée quand il y en a une ; sinon ces
  caractères sont remplacés par leur équivalent dactylographique, et le projet le note.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..concierge import implied_url
from ..config import setting
from . import OCR_PRODUCER

DPI = 200
TIMEOUT = 120.0
# En deçà de ce score, la « ligne » est un bout d'illustration que le détecteur a pris
# pour du texte.
MIN_SCORE = 0.3
# Le détecteur élargit ses boîtes autour des caractères : la hauteur du quadrilatère vaut
# à peu près l'interligne, le corps un peu moins. Seul le rapport entre lignes compte
# pour la mise en page, mais autant rester près de la vérité.
SIZE_RATIO = 0.75
# Une ligne au moins quatre fois plus large que haute donne un angle fiable.
LONG_LINE_RATIO = 4.0
# Deux lignes séparées d'au moins tant de hauteurs de ligne donnent une pente de marge fiable.
MARGIN_SPAN = 4.0
FONT_NAME = "VoxOCR"
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian, Ubuntu
    "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arch
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",  # Fedora
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
    "/Library/Fonts/Arial Unicode.ttf",
)
# Ce que la police de base ne sait pas écrire, et par quoi le remplacer.
SUBSTITUTIONS = str.maketrans(
    {"’": "'", "‘": "'", "“": '"', "”": '"', "…": "...", "—": "-", "–": "-", "œ": "oe", "Œ": "OE"}
)


class OCRError(RuntimeError):
    """Le service ne répond pas, ou refuse."""


def base_url(env: dict[str, str] | None = None) -> str:
    """L'adresse du service : réglée, ou déduite du conteneur du projet Compose."""
    if env is not None:
        return env.get("VOXLIBRIS_RAPIDOCR_BASE_URL", "").strip().rstrip("/")
    return setting("VOXLIBRIS_RAPIDOCR_BASE_URL").strip().rstrip("/") or implied_url("rapidocr")


def font_file() -> Path | None:
    """Une police TrueType complète, si la machine en a une."""
    chosen = setting("VOXLIBRIS_OCR_FONT").strip()
    candidates = [chosen] if chosen else list(FONT_CANDIDATES)
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


@dataclass(frozen=True)
class Line:
    """Une ligne lue : son texte, son quadrilatère en pixels, son score."""

    text: str
    box: tuple[tuple[float, float], ...]  # quatre sommets, à partir du coin haut gauche
    score: float = 1.0

    @property
    def height(self) -> float:
        (x0, y0), (x3, y3) = self.box[0], self.box[3]
        return math.hypot(x3 - x0, y3 - y0)

    @property
    def width(self) -> float:
        (x0, y0), (x1, y1) = self.box[0], self.box[1]
        return math.hypot(x1 - x0, y1 - y0)

    @property
    def angle(self) -> float:
        """Inclinaison de la ligne, en radians ; positive quand elle descend vers la
        droite. Moyenne des deux bords : chacun tremble un peu autour des caractères."""
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self.box
        return (math.atan2(y1 - y0, x1 - x0) + math.atan2(y2 - y3, x2 - x3)) / 2

    @property
    def left_middle(self) -> tuple[float, float]:
        (x0, y0), (x3, y3) = self.box[0], self.box[3]
        return (x0 + x3) / 2, (y0 + y3) / 2


class Client:
    """Le strict nécessaire pour parler au serveur, sans dépendance ajoutée."""

    def __init__(self, url: str = "", timeout: float = TIMEOUT) -> None:
        self.url = (url or base_url()).rstrip("/")
        self.timeout = timeout
        if not self.url:
            raise OCRError(
                "Aucun serveur RapidOCR : lancez la pile avec « --profile rapidocr », ou "
                "renseignez l'adresse d'un serveur dans les Réglages."
            )

    def _call(self, path: str, data: bytes | None = None, content_type: str = "") -> dict:
        request = urllib.request.Request(
            self.url + path, data=data, method="POST" if data else "GET"
        )
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise OCRError(f"RapidOCR refuse ({error.code}) : {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise OCRError(f"RapidOCR injoignable à {self.url} : {error}") from error

    def probe(self) -> dict:
        health = self._call("/health")
        if not health.get("ready"):
            raise OCRError("RapidOCR charge encore ses modèles : réessayez dans un instant.")
        return health

    def recognize(self, image: bytes) -> list[Line]:
        """Les lignes d'une image de page, telles que le serveur les rend."""
        payload = self._call("/ocr", image, "image/png")
        return [
            Line(
                str(entry.get("text", "")),
                tuple((float(x), float(y)) for x, y in entry["box"]),
                float(entry.get("score", 1.0)),
            )
            for entry in payload.get("lines", [])
            if len(entry.get("box") or ()) == 4
        ]


def render(page: pymupdf.Page, dpi: int = DPI) -> bytes:
    """L'image d'une page, en PNG, telle qu'on l'envoie au serveur."""
    return page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False).tobytes("png")


def skew(lines: list[Line]) -> float:
    """L'inclinaison de la page, en radians ; positive quand elle a tourné dans le sens
    des aiguilles d'une montre.

    Ce qui compte pour la mise en page, c'est que les lignes d'un même paragraphe
    retrouvent la même marge gauche. La pente de cette marge — médiane des pentes entre
    paires de longues lignes, insensible aux alinéas et aux titres centrés tant qu'ils
    restent minoritaires — est donc la mesure de référence. Les bords des boîtes du
    détecteur, un peu moins tournés que le texte qu'ils entourent, ne servent que faute
    de lignes en nombre.
    """
    long_lines = [line for line in lines if line.width >= LONG_LINE_RATIO * max(line.height, 1)]
    if not long_lines:
        return 0.0
    points = [line.left_middle for line in long_lines]
    span = MARGIN_SPAN * statistics.median(line.height for line in long_lines)
    slopes = [
        (x2 - x1) / (y2 - y1)
        for i, (x1, y1) in enumerate(points)
        for x2, y2 in points[i + 1 :]
        if abs(y2 - y1) >= span
    ]
    if len(slopes) >= 3:
        # Une marge verticale tournée d'un angle a vers la droite recule quand on descend.
        return -math.atan(statistics.median(slopes))
    return statistics.median(line.angle for line in long_lines)


def layer(page: pymupdf.Page, lines: list[Line], dpi: int = DPI, font: Path | None = None) -> int:
    """Pose sur la page, invisible, le texte lu à la place des mots. Renvoie le nombre
    de lignes posées.

    Les positions viennent de l'image rendue : elles sont dans le repère de la page telle
    qu'on la voit, celui où PyMuPDF place aussi le texte inséré, rotation comprise.
    """
    scale = 72.0 / dpi
    angle = skew(lines)
    cos, sin = math.cos(-angle), math.sin(-angle)
    cx, cy = page.rect.width / 2, page.rect.height / 2
    fontname = "helv"
    if font is not None:
        page.insert_font(fontname=FONT_NAME, fontfile=str(font))
        fontname = FONT_NAME
    measure = pymupdf.Font(fontfile=str(font)) if font is not None else pymupdf.Font("helv")

    placed = 0
    for line in lines:
        text = line.text.strip()
        if not text or line.score < MIN_SCORE:
            continue
        if font is None:
            text = text.translate(SUBSTITUTIONS)
        height = line.height * scale
        if height <= 0:
            continue
        x, y = line.left_middle
        x, y = x * scale, y * scale
        # Redressement autour du centre de la page, puis un retrait pour la marge que le
        # détecteur ajoute autour des caractères.
        dx, dy = x - cx, y - cy
        x, y = cx + dx * cos - dy * sin, cy + dx * sin + dy * cos
        size = max(height * SIZE_RATIO, 1.0)
        point = pymupdf.Point(x + size * 0.3, y + size * 0.35)
        # Étiré ou resserré pour couvrir exactement la largeur lue : la longueur d'une
        # ligne est ce qui dit, à l'extraction, qu'elle clôt un paragraphe. Le facteur
        # inverse en hauteur garde au texte son corps aux yeux de l'extraction.
        natural = measure.text_length(text, fontsize=size)
        wanted = max(line.width * scale - size * 0.6, 1.0)
        stretch = min(max(wanted / natural, 0.3), 4.0) if natural > 0 else 1.0
        morph = (point, pymupdf.Matrix(stretch, 0, 0, 1 / stretch, 0, 0))
        page.insert_text(point, text, fontsize=size, fontname=fontname, render_mode=3, morph=morph)
        placed += 1
    return placed


def ocr_pdf(
    source: Path,
    target: Path,
    client: Client,
    dpi: int = DPI,
    on_page: Callable[[int, int, int], None] | None = None,
) -> dict[str, object]:
    """Lit chaque page de `source` et écrit dans `target` le même PDF, couche de texte en
    plus. Renvoie ce qu'il y a à en dire : pages, lignes, durée, police."""
    started = time.monotonic()
    font = font_file()
    doc = pymupdf.open(source)
    total = 0
    for index, page in enumerate(doc):
        lines = client.recognize(render(page, dpi))
        total += layer(page, lines, dpi, font)
        if on_page:
            on_page(index + 1, len(doc), total)
    metadata = {k: v for k, v in (doc.metadata or {}).items() if v}
    metadata["producer"] = f"{OCR_PRODUCER} · RapidOCR"
    doc.set_metadata(metadata)
    if font is not None:
        # La police entière pèse plus que bien des scans : seuls les glyphes employés restent.
        doc.subset_fonts()
    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(target), garbage=3, deflate=True)
    pages = len(doc)
    doc.close()
    return {
        "pages": pages,
        "lines": total,
        "seconds": round(time.monotonic() - started, 1),
        "font": font.name if font else "police de base, caractères typographiques remplacés",
    }
