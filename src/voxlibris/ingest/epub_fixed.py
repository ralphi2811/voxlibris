"""EPUB à mise en page fixe : un PDF converti page à page par pdf2htmlEX.

Certains EPUB n'ont d'EPUB que l'enveloppe. Le fichier est une suite de pages
(`pageNum-17.html`, `pageNum-18.html`…), chacune reproduisant une page du PDF d'origine :
chaque ligne est un bloc positionné en absolu, chaque mot une suite de `span`, séparés
par des `span` vides qui règlent le crénage. Lu comme un EPUB ordinaire, ce balisage
donne « L ’ éd it ion ori g i na le » — un espace entre chaque fragment — et deux cents
« chapitres » d'une page.

Il faut donc le lire comme ce qu'il est, un PDF : recoller les fragments d'une ligne,
recomposer les paragraphes à partir des retraits et des interlignes, écarter les folios,
recoller la lettrine à son paragraphe, et retrouver les chapitres dans le sommaire, qui
lui est correctement écrit. Le reste — glyphes codés en zone privée — est l'affaire de
:mod:`.glyphs`.

Le convertisseur écrit la géométrie dans une feuille de style par page : `.x7 { left:
374px }`, `.y8 { bottom: 1446px }`, `.h4 { height: 60px }`. On la relit telle quelle.
"""

from __future__ import annotations

import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from bs4 import BeautifulSoup, NavigableString, Tag

from ..document import Chapter, Document, clean_title
from . import glyphs

# Au-delà de ce rapport à la hauteur du corps, une ligne est un titre ; au-delà du
# second, une lettre seule est une lettrine.
HEADING_RATIO = 1.25
DROP_CAP_RATIO = 1.8
# Retrait d'alinéa minimal, en fraction de la hauteur de ligne du corps.
INDENT_RATIO = 0.3
# Un interligne supérieur à ce multiple de l'interligne courant sépare deux paragraphes.
GAP_RATIO = 1.6
# Zones haute et basse de la page où logent folios et titres courants.
TOP_ZONE = 0.86
BOTTOM_ZONE = 0.14
SMALL_RATIO = 0.9
# En deçà, une page n'est qu'une illustration ou une garde.
MIN_PAGE_CHARS = 40
# Une police qui porte moins de cette fraction du texte est une police d'illustration :
# ses lignes courtes sont des étiquettes de dessin, des bulles, pas du récit.
RARE_FONT_SHARE = 0.05
MAX_LABEL_CHARS = 60
# À partir de ce nombre de mots restés illisibles, le livre est à relire.
MAX_BROKEN_WORDS = 5

RULE = re.compile(r"\.([A-Za-z_][\w-]*)\s*\{([^}]*)\}")
PX = re.compile(r"(left|bottom|height)\s*:\s*(-?[\d.]+)px")
FACE = re.compile(r"@font-face\s*\{[^}]*font-family:\s*(ff\d+)[^}]*url\(([^)]+)\)")
FONT_CLASS = re.compile(r"^ff\d+$")
LAYOUT = "pages fixes (pdf2htmlEX)"
VIEWPORT = re.compile(r"height\s*=\s*(\d+)")
DROP_CAP = re.compile(r"^[^\W\d_]['’]?$")
FOLIO = re.compile(r"^[\divxlcIVXLC]+$")
NUMBERED = re.compile(r"^\s*\d+\s*[.:)\-–—]?\s*$")


@dataclass
class Line:
    text: str
    x: float
    y: float  # distance du bas de la page, comme dans la feuille de style
    height: float
    page: int
    font: str = ""
    heading: bool = False
    running: bool = False


@dataclass
class Page:
    name: str
    lines: list[Line] = field(default_factory=list)
    height: float = 0.0

    @property
    def body(self) -> list[Line]:
        return [line for line in self.lines if not line.running and not line.heading]

    @property
    def chars(self) -> int:
        return sum(len(line.text) for line in self.lines if not line.running)


def is_fixed(soup: BeautifulSoup) -> bool:
    """Reconnaît une page produite par pdf2htmlEX."""
    return soup.select_one("#page-container div.t") is not None


def _rules(soup: BeautifulSoup) -> dict[str, dict[str, float]]:
    css = "\n".join(style.get_text() for style in soup.find_all("style"))
    rules: dict[str, dict[str, float]] = {}
    for match in RULE.finditer(css):
        props = {name: float(value) for name, value in PX.findall(match.group(2))}
        if props:
            rules.setdefault(match.group(1), {}).update(props)
    return rules


def _faces(soup: BeautifulSoup, page_name: str) -> dict[str, str]:
    """Classe de police → chemin du fichier de police dans le livre."""
    css = "\n".join(style.get_text() for style in soup.find_all("style"))
    base = posixpath.dirname(page_name)
    faces = {}
    for cls, url in FACE.findall(css):
        url = url.strip("'\" ")
        faces[cls] = posixpath.normpath(posixpath.join(base, url)) if base else url
    return faces


def _font_of(node: Tag | NavigableString, faces: dict[str, str]) -> str:
    for parent in node.parents:
        if not isinstance(parent, Tag):
            break
        for cls in parent.get("class") or ():
            if FONT_CLASS.match(cls):
                return faces.get(cls, "")
        if parent.name == "body":
            break
    return ""


def _measure(classes: list[str], rules: dict[str, dict[str, float]], prop: str) -> float | None:
    for cls in classes:
        value = rules.get(cls, {}).get(prop)
        if value is not None:
            return value
    return None


def parse_page(soup: BeautifulSoup, name: str, index: int, decoder: glyphs.Decoder) -> Page:
    """Relit les lignes positionnées d'une page, sans encore les interpréter."""
    rules = _rules(soup)
    faces = _faces(soup, name)
    meta = soup.find("meta", attrs={"name": "viewport"})
    height = 0.0
    if meta and (found := VIEWPORT.search(meta.get("content", ""))):
        height = float(found.group(1))

    raw: list[Line] = []
    for div in soup.select("div.t"):
        classes = list(div.get("class") or ())
        x = _measure(classes, rules, "left")
        y = _measure(classes, rules, "bottom")
        h = _measure(classes, rules, "height")
        if x is None or y is None or h is None:
            continue
        pieces = []
        fonts: Counter[str] = Counter()
        for node in div.descendants:
            if isinstance(node, NavigableString):
                font = _font_of(node, faces)
                pieces.append(decoder.decode(str(node), font))
                fonts[font] += len(str(node).strip())
        text = re.sub(r"[ \t\u00a0]+", " ", "".join(pieces)).strip()
        if text:
            raw.append(Line(text, x, y, h, index, fonts.most_common(1)[0][0]))

    raw.sort(key=lambda line: (-line.y, line.x))
    return Page(name, raw, height)


def _joiner(before: str, after: str, language: str) -> str:
    """Espace ou non entre deux fragments d'une ligne.

    Le convertisseur ne dit pas si un blanc sépare deux blocs. Le texte, lui, le dit
    souvent : une espace en fin de bloc, une ponctuation en tête. Sinon, le
    dictionnaire tranche — « inconfor » et « tables » ne sont des mots qu'ensemble.
    """
    if before.endswith(" ") or after[0] in "-,.;:!?»)":
        return ""
    head = re.search(r"[^\W\d_]+$", before)
    tail = re.match(r"[^\W\d_]+", after)
    if head and tail:
        whole = head.group() + tail.group()
        spell = glyphs.dictionary(language)
        parts = spell.known([head.group().lower(), tail.group().lower()])
        if len(parts) < 2 and spell.known([whole.lower()]):
            return ""
    return " "


def merge_baselines(page: Page, rare: set[str], language: str = "fr") -> None:
    """Recolle les fragments d'une même ligne.

    Une ligne dont la police change en cours de route — un mot à ligature, un tiret —
    est éclatée en plusieurs blocs sur la même ligne de base. Une étiquette de dessin
    posée à la même hauteur qu'une ligne du récit n'en fait pas partie : on ne recolle
    pas une police d'illustration à une police de texte, ni une lettrine à sa voisine.
    """
    lines: list[Line] = []
    for line in page.lines:
        previous = lines[-1] if lines else None
        if previous is not None:
            # Un tiret de coupure posé dans son propre bloc, parfois un rien plus bas
            # que sa ligne, appartient au mot qui le précède.
            if line.text == "-" and previous.text[-1:].isalpha():
                previous.text += "-"
                continue
            close = abs(previous.y - line.y) <= 0.15 * max(previous.height, line.height)
            similar = min(previous.height, line.height) > 0.6 * max(previous.height, line.height)
            kin = (previous.font in rare) == (line.font in rare)
            if close and similar and kin:
                joiner = _joiner(previous.text, line.text, language)
                previous.text = (previous.text + joiner + line.text).strip()
                previous.height = max(previous.height, line.height)
                continue
        lines.append(line)
    page.lines = lines


def prepare(pages: list[Page], language: str = "fr") -> tuple[float, int]:
    """Recolle, mesure et classe les lignes de toutes les pages.

    Renvoie la hauteur du corps et le nombre d'étiquettes d'illustration écartées.
    """
    rare = rare_fonts(pages)
    for page in pages:
        merge_baselines(page, rare, language)
    body = body_height(pages)
    return body, classify(pages, body, rare)


def body_height(pages: list[Page]) -> float:
    """Hauteur de ligne du texte courant : la plus répandue, pondérée par les caractères."""
    weights: Counter[float] = Counter()
    for page in pages:
        for line in page.lines:
            weights[round(line.height, 1)] += len(line.text)
    return weights.most_common(1)[0][0] if weights else 0.0


def rare_fonts(pages: list[Page]) -> set[str]:
    """Polices marginales : celles des dessins, des bulles, des ornements."""
    shares: Counter[str] = Counter()
    for page in pages:
        for line in page.lines:
            shares[line.font] += len(line.text)
    total = sum(shares.values()) or 1
    return {font for font, chars in shares.items() if chars < RARE_FONT_SHARE * total}


def _normalized(text: str) -> str:
    return re.sub(r"[^A-ZÀ-Ý]+", " ", text.upper()).strip()


def _lettering(text: str) -> bool:
    """Lettres éparses d'une illustration : « K r o k … m o u », « O R H T »."""
    tokens = text.split()
    singles = sum(1 for t in tokens if len(t) == 1)
    return len(tokens) >= 3 and singles >= 0.75 * len(tokens)


def classify(pages: list[Page], body: float, rare: set[str] | None = None) -> int:
    """Marque titres et éléments courants, une fois la hauteur du corps connue.

    Un titre courant se reconnaît à sa place — en marge haute ou basse — et à sa
    composition : plus petit que le corps, ou en capitales, ou réduit à une initiale
    ornée. Le plus sûr reste qu'il se répète : le même texte en tête de trois pages
    n'est pas du récit. Renvoie le nombre d'étiquettes d'illustration écartées.
    """
    rare = rare or set()
    labels = 0
    seen: Counter[str] = Counter()
    edges: dict[int, bool] = {}
    for page in pages:
        for line in page.lines:
            edge = bool(page.height) and (
                line.y > TOP_ZONE * page.height or line.y < BOTTOM_ZONE * page.height
            )
            edges[id(line)] = edge
            if edge:
                seen[_normalized(line.text)] += 1

    for page in pages:
        for line in page.lines:
            text = line.text
            edge = edges[id(line)]
            small = line.height < SMALL_RATIO * body
            letters = any(ch.isalpha() for ch in text)
            folio = bool(FOLIO.match(text)) and (text.isdigit() or edge)
            caps = letters and text == text.upper()
            repeated = edge and seen[_normalized(text)] >= 3 and letters
            # Une ligne d'un ou deux caractères n'est du texte que si c'est une lettrine.
            tiny = len(text) <= 2 and line.height < DROP_CAP_RATIO * body
            label = (
                line.font in rare
                and len(text) <= MAX_LABEL_CHARS
                and line.height < HEADING_RATIO * body
            )
            line.running = (
                folio
                or tiny
                or not any(ch.isalnum() for ch in text)
                or _lettering(text)
                or repeated
                or label
                or (edge and (small or caps))
            )
            labels += label and not (folio or repeated)
            line.heading = (
                not line.running
                and line.height >= HEADING_RATIO * body
                and not DROP_CAP.match(text)
            )
    return labels


def _is_drop_cap(line: Line, body: float) -> bool:
    return line.height >= DROP_CAP_RATIO * body and bool(DROP_CAP.match(line.text))


def paragraphs(pages: list[Page], body: float) -> list[str]:
    """Recompose les paragraphes d'une suite de pages."""
    built: list[list[str]] = []
    previous: Line | None = None

    for page in pages:
        lines = [line for line in page.lines if not line.running]
        caps = [line for line in lines if _is_drop_cap(line, body)]
        lines = [line for line in lines if line not in caps]
        anchors = [line.x for line in page.body if line not in caps] or [line.x for line in lines]
        margin = min(anchors) if anchors else 0.0
        steps = [a.y - b.y for a, b in zip(lines, lines[1:], strict=False) if a.y > b.y]
        pitch = median(steps) if steps else body * 1.2
        # La lettrine se lit avec les lignes qu'elle borde, quel que soit leur ordre.
        pending = {id(cap): cap for cap in caps}
        last_cap: Line | None = None
        previous_fresh = True

        for line in lines:
            beside = next(
                (cap for cap in caps if cap.y - 2 <= line.y < cap.y + cap.height + 2), None
            )
            text = line.text
            if beside is not None and id(beside) in pending:
                del pending[id(beside)]
                text = beside.text + text
                fresh = True
            elif beside is not None and beside is last_cap:
                fresh = False
            else:
                same_page = previous is not None and previous.page == line.page
                gap = same_page and previous.y - line.y > GAP_RATIO * pitch
                indented = line.x > margin + INDENT_RATIO * body
                # Un texte qui contourne une image se décale d'un bloc : chaque ligne
                # est en retrait, mais aligne la précédente, qui n'ouvrait rien.
                shifted = same_page and not previous_fresh and abs(previous.x - line.x) < 0.2 * body
                # Une minuscule n'ouvre pas de paragraphe — sauf sous un titre.
                lower = text[:1].islower()
                fresh = (
                    previous is None
                    or line.heading
                    or previous.heading
                    or gap
                    or text.startswith(("—", "–", "«"))
                    or (indented and not shifted and not lower)
                )
            last_cap = beside
            previous_fresh = fresh
            if fresh or not built:
                built.append([text])
            else:
                built[-1].append(text)
            previous = line

    joined = []
    for parts in built:
        merged = ""
        for part in parts:
            if merged.endswith("-") and (
                part[:1].islower() or (merged[-2:-1].isupper() and part[:1].isupper())
            ):
                merged = merged[:-1] + part
            elif merged:
                merged += " " + part
            else:
                merged = part
        joined.append(merged)
    return joined


def _spine_names(book) -> list[str]:
    import ebooklib

    names = []
    for spine_id, _ in book.spine:
        item = book.get_item_with_id(spine_id)
        if item is not None and item.get_type() == ebooklib.ITEM_DOCUMENT:
            names.append(item.get_name())
    return names


def chapter_starts(names: list[str], toc: dict[str, str]) -> list[tuple[int, str]]:
    """Positions et titres des chapitres d'après le sommaire.

    Le sommaire d'un tel EPUB liste d'abord les chapitres, puis une entrée par page —
    intitulée de son seul numéro. Ces dernières ne nomment rien : on les ignore.
    """
    starts = []
    for index, name in enumerate(names):
        title = toc.get(name) or next((t for href, t in toc.items() if name.endswith(href)), "")
        if title and not NUMBERED.match(title):
            starts.append((index, title))
    return starts


def _heading_starts(pages: list[Page]) -> list[tuple[int, str]]:
    """Sans sommaire : une page qui s'ouvre sur un titre commence un chapitre."""
    starts = []
    for index, page in enumerate(pages):
        visible = [line for line in page.lines if not line.running]
        if visible and visible[0].heading and page.chars >= MIN_PAGE_CHARS:
            starts.append((index, visible[0].text))
    return starts


def _strip_title(pages: list[Page], title: str) -> str:
    """Retire les lignes de titre en tête d'un chapitre et renvoie l'intitulé lu."""
    for page in pages:
        visible = [line for line in page.lines if not line.running]
        if not visible:
            continue
        read = []
        for line in visible:
            if not line.heading:
                break
            read.append(line.text)
            page.lines.remove(line)
        return " ".join(read) if read else ""
    return ""


def ingest(
    book, path: Path, title: str = "", author: str = "", toc: dict[str, str] | None = None
) -> Document:
    """Extrait un document depuis un EPUB à mise en page fixe déjà ouvert."""
    import ebooklib

    from .epub import _metadata, _toc_titles

    toc = toc if toc is not None else _toc_titles(book)
    fonts = {
        item.get_name(): item.get_content() for item in book.get_items_of_type(ebooklib.ITEM_FONT)
    }
    decoder = glyphs.Decoder(fonts)

    names = _spine_names(book)
    pages: list[Page] = []
    for index, name in enumerate(names):
        item = book.get_item_with_href(name)
        # `get_content()` réécrit le document et perd sa feuille de style : on lit le
        # fichier tel qu'il est dans l'archive.
        soup = BeautifulSoup(item.content, "html.parser")
        pages.append(parse_page(soup, name, index, decoder))

    language = _metadata(book, "language", "fr")[:2] or "fr"
    body, labels = prepare(pages, language)

    starts = chapter_starts(names, toc)
    if len(starts) < 2:
        starts = _heading_starts(pages) or [(0, title or _metadata(book, "title"))]
    # Un chapitre commence à sa page : ce qui précède la première est liminaire.
    bounds = [
        (s, e, t) for (s, t), (e, _) in zip(starts, starts[1:] + [(len(pages), "")], strict=False)
    ]

    chapters: list[Chapter] = []
    blank = 0
    texts: list[str] = []
    for first, last, listed in bounds:
        span = [page for page in pages[first:last]]
        kept = [page for page in span if page.chars >= MIN_PAGE_CHARS]
        blank += len(span) - len(kept)
        read = _strip_title(kept, listed)
        text = paragraphs(kept, body)
        if not text:
            continue
        chapters.append(
            Chapter(
                number=len(chapters) + 1,
                title=clean_title(listed or read),
                paragraphs=text,
                source_pages=(first + 1, last),
            )
        )
        texts.extend(text)

    resolved = decoder.resolve(texts, language)
    if resolved:
        for chapter in chapters:
            chapter.paragraphs = [glyphs.apply(p, resolved) for p in chapter.paragraphs]
    report = decoder.describe(resolved)
    broken = sum(p.count(glyphs.UNRESOLVED) for c in chapters for p in c.paragraphs)

    notes: dict[str, object] = {"layout": LAYOUT}
    if starts and starts[0][0]:
        notes["skipped"] = [f"{starts[0][0]} pages liminaires avant « {starts[0][1]} »"]
    if blank:
        notes.setdefault("skipped", []).append(f"{blank} pages sans texte (illustrations)")
    if labels:
        notes.setdefault("skipped", []).append(f"{labels} étiquettes de dessins")
    if report["decoded"] or report["voted"] or report["unresolved"]:
        notes["glyphes"] = (
            f"{report['decoded']} glyphes privés décodés par les contours, "
            f"{report['voted']} par le dictionnaire, {report['unresolved']} non résolus"
        )
    if not decoder.available:
        notes["glyphes"] = "fonttools absent : glyphes privés non décodés"

    return Document(
        title=title or _metadata(book, "title") or path.stem,
        author=author or _metadata(book, "creator", "Inconnu"),
        year=(_metadata(book, "date") or None),
        language=language,
        chapters=chapters,
        source=path,
        # Le texte d'un PDF converti est fiable ; quelques glyphes restés sans lecture
        # se voient dans le texte, et la relecture les relèvera si on la demande.
        needs_review=broken >= MAX_BROKEN_WORDS,
        notes=notes,
    )
