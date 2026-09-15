"""Extraction du texte d'un PDF scanné portant une couche OCR invisible.

Les scans d'Internet Archive et de Google Books contiennent une couche de texte produite
par OCR, calée sur l'image. Elle conserve la position et la taille de chaque mot, ce qui
permet un nettoyage bien plus sûr que des expressions régulières sur du texte à plat :
les ornements de page s'éliminent par leur géométrie, et les fragments d'illustration mal
reconnus par leur taille aberrante.

Toutes les valeurs de mise en page proviennent de `layout.detect` — rien n'est codé en
dur, contrairement à la première version de ce code.

Le texte produit **doit être relu** : l'OCR confond `il` et `1l`, avale des mots, coupe
des phrases entre deux pages. Ces erreurs sont invisibles à l'œil dans un fichier texte,
mais s'entendent immédiatement une fois lues à voix haute.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import pymupdf

from ..document import Chapter, Document, clean_title
from .layout import Layout, body_page_range, detect

LINE_TOLERANCE = 3.0  # écart vertical maximal, en points, entre spans d'une même ligne
INDENT = 4.0  # décalage minimal, en points, pour qu'une ligne compte comme un alinéa
# Une ligne plus courte que cette fraction d'une ligne pleine, et qui s'achève sur une
# ponctuation finale, est la dernière d'un paragraphe — même sans alinéa après elle.
SHORT_LINE = 0.8
CLOSING = tuple('.!?…»”"’')
MIN_LETTERS = 3
MIN_WORD_RATIO = 0.6
LONG_LINE = 20  # au-delà, une ligne est du texte même si l'OCR l'a massacrée
VOWELS = set("aeiouyàâäéèêëîïôöûüùAEIOUYÀÂÄÉÈÊËÎÏÔÖÛÜÙ")
PUNCT = ".,;:!?«»()[]\"'‘’“”‚…—–-*"

CHAPTER_MARKER = re.compile(r"\b(?:CHAPITRE|CHAPTER|CAPÍTULO|KAPITEL)\s*(\d{1,3}|[IVXLC]+)\b", re.I)
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


@dataclass
class Line:
    text: str
    y: float
    x: float
    size: float
    page: int
    width: float = (
        0.0  # inconnue (zéro) sur les anciens appels : la règle de la ligne courte ne joue pas
    )


@dataclass
class Page:
    number: int
    header: str
    lines: list[Line]
    dropped: list[str] = field(default_factory=list)
    chapter: int | None = None


def _roman_to_int(value: str) -> int:
    total = previous = 0
    for char in reversed(value.upper()):
        current = ROMAN.get(char, 0)
        total += -current if current < previous else current
        previous = max(previous, current)
    return total


def _chapter_number(text: str) -> int | None:
    match = CHAPTER_MARKER.search(text)
    if not match:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else _roman_to_int(token)


def normalize_header(raw: str) -> str:
    """Réduit un en-tête à sa partie signifiante, ornements typographiques ôtés."""

    def significant(word: str) -> bool:
        if word.isdigit():
            return True
        return len(word) >= 2 and word.upper() == word and any(c.isalpha() for c in word)

    kept = [word for word in raw.split() if significant(word)]
    cleaned = re.sub(r"[^A-ZÀ-ÝŒÇ'’\s\d-]+", " ", " ".join(kept))
    return re.sub(r"\s+", " ", cleaned).strip()


def is_garbage(text: str, size: float, layout: Layout) -> bool:
    """Vrai si la ligne provient d'une illustration ou d'un artefact d'OCR.

    Le critère porte sur les *mots* et non sur les caractères : une fin de paragraphe
    comme « stock ! »). » n'a que 45 % de lettres mais reste du texte parfaitement
    valide, alors que « L% Æ » ou « 107 70 » ne contiennent aucun mot.
    """
    if not (layout.body_size_min <= size <= layout.body_size_max):
        return True
    stripped = text.strip()
    if CHAPTER_MARKER.search(stripped):  # en-tête ayant débordé de sa bande
        return True
    # Le charabia d'illustration est toujours bref. Une ligne longue à la bonne taille
    # est du texte, même massacrée par l'OCR, et doit être conservée pour être corrigée
    # à la relecture plutôt que perdue ici.
    if len(stripped) >= LONG_LINE:
        return False
    if sum(c.isalpha() for c in stripped) < MIN_LETTERS:
        return True
    # Les capitales isolées sont des fragments d'illustration : le corps du texte
    # n'emploie pas de mots courts tout en majuscules.
    if stripped.upper() == stripped and len(stripped) < 8:
        return True

    word_like = counted = 0
    for token in stripped.split():
        core = token.strip(PUNCT)
        if not core:
            continue
        counted += 1
        letters = [c for c in core if c.isalpha() or c in "'’-"]
        is_word = len(letters) == len(core) and any(c in VOWELS for c in core)
        # Une lettre isolée n'est un mot qu'en de rares cas (« à », « a », « y »).
        if is_word and (len(core) >= 2 or core.lower() in {"a", "à", "y"}):
            word_like += 1

    return counted == 0 or word_like / counted < MIN_WORD_RATIO


def group_lines(page: pymupdf.Page, layout: Layout) -> tuple[str, list[Line], list[str]]:
    """Regroupe les spans d'une page en lignes, séparant en-tête, corps et rebut."""
    header_spans: list[tuple[float, str]] = []
    body_spans: list[tuple[float, float, float, str, float]] = []

    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                y = span["bbox"][1]
                if y < layout.header_y:
                    header_spans.append((span["bbox"][0], text))
                elif y < layout.footer_y:
                    body_spans.append((y, span["bbox"][0], span["size"], text, span["bbox"][2]))

    header = normalize_header(" ".join(t for _, t in sorted(header_spans)))

    body_spans.sort(key=lambda s: (s[0], s[1]))
    lines: list[Line] = []
    dropped: list[str] = []
    current: list[tuple[float, float, float, str, float]] = []

    def flush() -> None:
        if not current:
            return
        current.sort(key=lambda s: s[1])
        text = " ".join(s[3] for s in current)
        size = statistics.median(s[2] for s in current)
        if is_garbage(text, size, layout):
            dropped.append(f"[{size:4.1f}pt] {text}")
        else:
            width = max(s[4] for s in current) - current[0][1]
            lines.append(Line(text, current[0][0], current[0][1], size, page.number + 1, width))
        current.clear()

    for span in body_spans:
        if current and abs(span[0] - current[-1][0]) > LINE_TOLERANCE:
            flush()
        current.append(span)
    flush()

    return header, lines, dropped


def dehyphenate(lines: list[str]) -> list[str]:
    """Recolle les mots coupés en fin de ligne (« heu- » + « reux » → « heureux »)."""
    out: list[str] = []
    for line in lines:
        if out and out[-1].endswith("-"):
            out[-1] = out[-1][:-1] + line.lstrip()
        else:
            out.append(line)
    return out


def build_paragraphs(lines: list[Line]) -> list[str]:
    """Reconstruit les paragraphes à partir des alinéas et des tirets de dialogue.

    La marge gauche diffère entre recto et verso : elle est mesurée page par page, sans
    quoi toutes les lignes des rectos passent pour des alinéas et chaque ligne devient un
    paragraphe.

    Bien des éditions n'ont pas d'alinéa : le paragraphe ne se voit qu'à sa dernière ligne,
    plus courte et close par une ponctuation. La largeur d'une ligne pleine est la médiane
    des largeurs de la page — sur une page de récit, la plupart des lignes sont pleines.
    """
    if not lines:
        return []
    margins: dict[int, float] = {}
    widths: dict[int, list[float]] = {}
    for line in lines:
        margins[line.page] = min(margins.get(line.page, line.x), line.x)
        if line.width:
            widths.setdefault(line.page, []).append(line.width)
    full = {page: statistics.median(found) for page, found in widths.items()}

    paragraphs: list[list[str]] = []
    closed = False
    for line in lines:
        text = line.text.strip()
        starts_dialogue = text.startswith(("—", "–", "-"))
        indented = line.x > margins[line.page] + INDENT
        if not paragraphs or starts_dialogue or indented or closed:
            paragraphs.append([text])
        else:
            paragraphs[-1].append(text)
        closed = (
            line.page in full
            and 0 < line.width < SHORT_LINE * full[line.page]
            and text.endswith(CLOSING)
        )
    return [" ".join(dehyphenate(p)) for p in paragraphs]


def assign_chapters(pages: list[Page]) -> dict[int, str]:
    """Attribue un chapitre à chaque page et déduit les titres depuis les en-têtes.

    Beaucoup d'éditions impriment « CHAPITRE N » sur les versos et le titre du chapitre
    sur les rectos. Ces en-têtes alternés fournissent un chapitrage bien plus fiable que
    la recherche de titres dans le corps du texte.
    """
    anchors: list[tuple[int, int]] = []
    for index, page in enumerate(pages):
        number = _chapter_number(page.header)
        if number is not None:
            page.chapter = number
            anchors.append((index, number))

    if not anchors:
        return {}

    # Les en-têtes alternent d'une page à l'autre : une page portant un titre est donc
    # presque toujours voisine d'une page « CHAPITRE N », qui lui donne son chapitre.
    titles: dict[int, list[str]] = {}
    for index, page in enumerate(pages):
        if not page.header or _chapter_number(page.header) is not None:
            continue
        for neighbour in (index - 1, index + 1):
            if 0 <= neighbour < len(pages) and pages[neighbour].chapter is not None:
                titles.setdefault(pages[neighbour].chapter, []).append(page.header)
                break

    def title_of(chapter: int) -> str | None:
        seen = titles.get(chapter)
        return statistics.mode(seen) if seen else None

    for page in pages[: anchors[0][0]]:
        page.chapter = anchors[0][1]
    for page in pages[anchors[-1][0] :]:
        page.chapter = anchors[-1][1]

    for (i, chapter), (j, following) in zip(anchors, anchors[1:], strict=False):
        gap = pages[i + 1 : j]
        if chapter == following:
            for page in gap:
                page.chapter = chapter
            continue
        current = title_of(chapter)
        # On remonte depuis l'ancre suivante : la rupture est la première page d'une
        # série finale qui ne porte plus l'en-tête du chapitre courant.
        split = len(gap)
        for k in range(len(gap) - 1, -1, -1):
            if gap[k].header and current and gap[k].header == current:
                break
            split = k
        for k, page in enumerate(gap):
            page.chapter = chapter if k < split else following

    return {c: title_of(c) or "" for c in sorted({p.chapter for p in pages if p.chapter})}


def trim_back_matter(pages: list[Page], titles: dict[int, str]) -> list[str]:
    """Retire les pages d'annexes que la détection de plage a laissé passer.

    Table des matières, notice biographique et catalogue d'éditeur sont denses en texte :
    la mesure du corps les prend pour du récit. Mais leur titre courant les trahit — il
    n'est ni « CHAPITRE N » ni le titre du chapitre en cours.

    Ces pages coûtent cher si on les laisse : leur reconnaissance de caractères est
    particulièrement mauvaise, et le moteur s'échine ensuite à lire à voix haute des
    lignes telles que « #sixQ sup, nu.f st bit s alta cent huit ». On rogne donc depuis
    la fin, en s'arrêtant à la première page qui ne trahit rien — une page sans en-tête
    est du récit, comme l'est une page d'ouverture de chapitre.
    """
    known = {t.upper() for t in titles.values() if t}
    dropped: list[str] = []
    while pages:
        header = pages[-1].header
        if not header or _chapter_number(header) is not None or header.upper() in known:
            break
        dropped.append(f"p.{pages[-1].number} « {header} »")
        pages.pop()
    return list(reversed(dropped))


def ingest(
    path,
    first: int | None = None,
    last: int | None = None,
    title: str = "",
    author: str = "Inconnu",
) -> Document:
    """Extrait un document depuis un PDF scanné doté d'une couche OCR."""
    doc = pymupdf.open(path)
    layout = detect(doc)
    auto_first, auto_last = body_page_range(doc, layout)
    first = first or auto_first
    last = last or auto_last

    pages: list[Page] = []
    for number in range(first - 1, last):
        header, lines, dropped = group_lines(doc[number], layout)
        pages.append(Page(number + 1, header, lines, dropped))

    chapter_titles = assign_chapters(pages)
    back_matter = trim_back_matter(pages, chapter_titles)
    dropped_report = [f"p.{page.number:>3} {line}" for page in pages for line in page.dropped]

    if not chapter_titles:
        # Aucun en-tête exploitable : tout le corps forme un chapitre unique, à
        # redécouper dans l'interface.
        paragraphs = build_paragraphs([line for page in pages for line in page.lines])
        chapters = [Chapter(1, title or "Texte intégral", paragraphs, (first, last))]
    else:
        chapters = []
        for number, raw_title in chapter_titles.items():
            owned = [p for p in pages if p.chapter == number]
            paragraphs = build_paragraphs([line for page in owned for line in page.lines])
            display = clean_title(raw_title) or f"Chapitre {number}"
            # Le titre imprimé sur la page d'ouverture figure déjà dans l'en-tête : il
            # n'a pas à être lu deux fois.
            if paragraphs and paragraphs[0].strip().lower() == display.lower():
                paragraphs.pop(0)
            chapters.append(
                Chapter(number, display, paragraphs, (owned[0].number, owned[-1].number))
            )

    return Document(
        title=title or getattr(path, "stem", "Sans titre"),
        author=author,
        chapters=chapters,
        source=path,
        needs_review=True,
        notes={
            "layout": layout.describe(),
            "pages": f"{first}-{pages[-1].number if pages else last}",
            "back_matter": back_matter,
            "dropped_lines": dropped_report,
        },
    )
