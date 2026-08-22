"""Mesure la mise en page d'un PDF au lieu de la supposer.

La première version de ce code portait en dur les valeurs relevées à la main sur un
livre : bande d'en-tête à 22 pt, pied à 425, corps de texte entre 8,8 et 14,5 pt. Ces
nombres n'ont aucun sens pour un autre ouvrage — ils dépendent du format de la page et
de la fonte de l'édition.

Tout se déduit pourtant du document lui-même :

- **Le corps de texte est ce qu'il y a de plus abondant.** La taille de police dominante,
  pondérée par le nombre de caractères, est celle du texte courant ; titres et notes sont
  minoritaires par construction.
- **Les éléments courants sont composés plus petit que le corps.** Titre courant et folio
  sont en petites capitales ou en corps réduit — c'est une constante de la typographie du
  livre. La frontière se place alors entre la plus basse de ces petites lignes et la plus
  haute ligne de texte courant.

Une première version cherchait plutôt les hauteurs *récurrentes* d'une page à l'autre.
L'idée est juste sur un PDF natif, mais elle échoue sur un scan : l'OCR fait varier la
hauteur d'un même titre courant de plusieurs points selon les pages, et plus aucune
hauteur ne se répète exactement.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pymupdf

# Les éléments courants se logent dans ces fractions haute et basse de la page ; une
# petite ligne au milieu serait une note ou une légende, pas un ornement de page.
TOP_ZONE = 0.14
BOTTOM_ZONE = 0.86
# En deçà de cette fraction du corps, une ligne est tenue pour un élément courant.
SMALL_RATIO = 0.85
MARGIN = 2.0  # points de garde de part et d'autre de la frontière calculée


@dataclass(frozen=True)
class Layout:
    """Géométrie mesurée d'un PDF."""

    page_height: float
    header_y: float
    footer_y: float
    body_size: float
    body_size_min: float
    body_size_max: float
    margin_left: float

    def is_body(self, y: float, size: float) -> bool:
        in_band = self.header_y <= y < self.footer_y
        return in_band and self.body_size_min <= size <= self.body_size_max

    def describe(self) -> str:
        return (
            f"page {self.page_height:.0f} pt, en-tête < {self.header_y:.0f}, "
            f"pied > {self.footer_y:.0f}, corps {self.body_size:.1f} pt "
            f"({self.body_size_min:.1f}–{self.body_size_max:.1f})"
        )


def _line_positions(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    """Renvoie (y, x, taille médiane) pour chaque ligne de la page."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            sizes = sorted(s["size"] for s in spans)
            lines.append((line["bbox"][1], line["bbox"][0], sizes[len(sizes) // 2]))
    return lines


def detect(doc: pymupdf.Document, sample: int = 60) -> Layout:
    """Déduit la mise en page en échantillonnant les pages du document."""
    pages = list(range(len(doc)))
    if len(pages) > sample:
        # Échantillon régulier : le liminaire et les annexes ne ressemblent pas au corps,
        # les prendre tous fausserait la mesure.
        step = len(pages) / sample
        pages = [int(i * step) for i in range(sample)]

    height = doc[0].rect.height
    sizes: Counter[float] = Counter()
    per_page: list[list[tuple[float, float, float]]] = []

    for number in pages:
        positions = _line_positions(doc[number])
        if not positions:
            continue
        per_page.append(positions)
        for _, _, size in positions:
            sizes[round(size, 1)] += 1

    if not per_page:
        raise ValueError("Aucun texte trouvé : ce PDF a-t-il une couche de texte ?")

    body_size = _dominant_size(sizes)
    small = body_size * SMALL_RATIO
    body_min, body_max = body_size * 0.78, body_size * 1.32

    running_top: list[float] = []
    running_bottom: list[float] = []
    body_top: list[float] = []
    body_bottom: list[float] = []
    lefts: list[float] = []

    for positions in per_page:
        body = [(y, x) for y, x, size in positions if body_min <= size <= body_max]
        if body:
            body_top.append(min(y for y, _ in body))
            body_bottom.append(max(y for y, _ in body))
            lefts += [x for _, x in body]
        for y, _, size in positions:
            if size >= small:
                continue
            if y < height * TOP_ZONE:
                running_top.append(y)
            elif y > height * BOTTOM_ZONE:
                running_bottom.append(y)

    header_y = _boundary(running_top, body_top, above=True)
    footer_y = _boundary(running_bottom, body_bottom, above=False)
    # Garde-fou : quoi qu'aient donné les mesures, les bandes ne débordent jamais des
    # zones où un ornement de page peut se trouver. Mieux vaut laisser passer un titre
    # courant que d'amputer le texte.
    header_y = min(header_y, height * TOP_ZONE) if header_y is not None else 0.0
    footer_y = max(footer_y, height * BOTTOM_ZONE) if footer_y is not None else height
    lefts.sort()

    return Layout(
        page_height=height,
        header_y=header_y,
        footer_y=footer_y,
        body_size=body_size,
        # Les tailles relevées sur un scan fluctuent d'une ligne à l'autre : la
        # fourchette doit être large, sans quoi du texte serait pris pour un titre.
        body_size_min=body_min,
        body_size_max=body_max,
        margin_left=lefts[len(lefts) // 10] if lefts else 0.0,
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def _boundary(running: list[float], body: list[float], above: bool) -> float | None:
    """Place la frontière à mi-chemin entre les éléments courants et le texte.

    Les deux côtés sont résumés par leur médiane, et non par un extrême. Chaque
    population contient en effet des valeurs aberrantes qui se ressemblent : les pages
    d'ouverture de chapitre portent leur titre courant bien plus bas que les autres, et,
    symétriquement, l'OCR attribue parfois à un titre courant une taille qui tombe dans
    la fourchette du corps, le faisant passer pour la première ligne de texte. Un
    maximum d'un côté ou un minimum de l'autre suffirait à faire mordre la bande sur le
    texte, ou à ne rien couper du tout.
    """
    if not running or not body:
        return None
    edge, wall = _percentile(running, 0.50), _percentile(body, 0.50)
    if above:
        return (edge + wall) / 2 if edge < wall else None
    return (edge + wall) / 2 if edge > wall else None


def _dominant_size(sizes: Counter[float]) -> float:
    """Taille de police du corps de texte, en caractères comptés.

    La médiane pondérée est préférée au mode : sur un scan, l'OCR attribue à chaque mot
    une taille légèrement différente, et le mode tomberait sur une valeur arbitraire
    parmi des dizaines presque équivalentes.
    """
    if not sizes:
        raise ValueError("Aucune taille de police relevée")
    ordered = sorted(sizes.items())
    total = sum(sizes.values())
    seen = 0
    for size, count in ordered:
        seen += count
        if seen >= total / 2:
            return size
    return ordered[-1][0]


def body_page_range(doc: pymupdf.Document, layout: Layout, gap: int = 2) -> tuple[int, int]:
    """Devine les premières et dernières pages de corps, liminaire et annexes exclus.

    On cherche la plus longue **plage contiguë** de pages denses, et non simplement la
    première et la dernière page dense. Une page de garde ou une publicité d'éditeur
    peut être bavarde et se trouver isolée en tête ou en queue d'ouvrage ; exiger la
    contiguïté l'écarte, là qu'un simple seuil la retiendrait.

    `gap` autorise quelques pages maigres consécutives à l'intérieur de la plage : une
    pleine page d'illustration en plein récit ne doit pas la scinder.

    Le résultat reste une estimation, à confirmer par l'utilisateur : aucune heuristique
    ne distingue de façon sûre une préface d'un premier chapitre.
    """
    counts = [_body_chars(doc[number], layout) for number in range(len(doc))]
    if not counts:
        return 1, len(doc)
    floor = max(60, sorted(counts)[len(counts) // 2] * 0.35)

    best: tuple[int, int] | None = None
    start: int | None = None
    last_dense = 0
    for index, count in enumerate(counts):
        if count >= floor:
            if start is None:
                start = index
            last_dense = index
        elif start is not None and index - last_dense > gap:
            if best is None or last_dense - start > best[1] - best[0]:
                best = (start, last_dense)
            start = None
    if start is not None and (best is None or last_dense - start > best[1] - best[0]):
        best = (start, last_dense)

    if best is None:
        return 1, len(doc)
    return best[0] + 1, best[1] + 1


def _body_chars(page: pymupdf.Page, layout: Layout) -> int:
    total = 0
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if text and layout.is_body(span["bbox"][1], span["size"]):
                    total += len(text)
    return total
