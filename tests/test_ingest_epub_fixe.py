"""Tests de l'ingestion d'un EPUB à mise en page fixe (PDF converti par pdf2htmlEX).

Ces fichiers ressemblent à des EPUB et se lisent comme des PDF : lignes positionnées,
mots éclatés en fragments, folios, lettrines, glyphes en zone privée. Chaque règle du
lecteur répond à un défaut observé sur un vrai livre, et chacune a son test.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from voxlibris.document import clean_title
from voxlibris.ingest import epub as E
from voxlibris.ingest import epub_fixed as F
from voxlibris.ingest import glyphs

# --- polices ---------------------------------------------------------------------

SQUARE = [(0, 0), (500, 0), (500, 500), (0, 500)]
TRIANGLE = [(0, 0), (500, 0), (250, 700)]
DIAMOND = [(250, 0), (500, 350), (250, 700), (0, 350)]
PENTAGON = [(250, 0), (500, 200), (400, 700), (100, 700), (0, 200)]


def build_font(cmap: dict[int, str], outlines: dict[str, list]) -> bytes:
    """Fabrique une police TrueType minimale : un contour par glyphe."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    names = [".notdef", *outlines]
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(names)
    fb.setupCharacterMap(cmap)
    drawn = {}
    for name in names:
        pen = TTGlyphPen(None)
        if outlines.get(name):
            points = outlines[name]
            pen.moveTo(points[0])
            for point in points[1:]:
                pen.lineTo(point)
            pen.closePath()
        drawn[name] = pen.glyph()
    fb.setupGlyf(drawn)
    fb.setupHorizontalMetrics({name: (600, 0) for name in names})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "Essai", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    out = io.BytesIO()
    fb.save(out)
    return out.getvalue()


@pytest.fixture(scope="module")
def fonts() -> dict[str, bytes]:
    """Une police Unicode et sa jumelle fautive, aux mêmes dessins.

    Dans la jumelle, « e » et « n » sont codés en zone privée ; le « fi » n'existe que
    là, sans équivalent Unicode nulle part.
    """
    unicode_font = build_font(
        {ord("e"): "e", ord("n"): "n", ord(" "): "space"},
        {"e": SQUARE, "n": TRIANGLE, "space": []},
    )
    private_font = build_font(
        {0xE048: "uniE048", 0xE051: "uniE051", 0xE0DF: "uniE0DF", 0xE003: "uniE003"},
        {"uniE048": SQUARE, "uniE051": TRIANGLE, "uniE0DF": DIAMOND, "uniE003": []},
    )
    return {"fonts/a.ttf": unicode_font, "fonts/b.ttf": private_font}


class TestGlyphes:
    def test_les_contours_identiques_donnent_le_caractere(self, fonts):
        decoder = glyphs.Decoder(fonts)
        assert decoder.decode("", "fonts/b.ttf") == "en"

    def test_un_glyphe_sans_dessin_est_une_espace(self, fonts):
        decoder = glyphs.Decoder(fonts)
        assert decoder.decode("", "fonts/b.ttf") == "e e"

    def test_un_glyphe_sans_jumeau_recoit_un_substitut_puis_le_dictionnaire_tranche(self, fonts):
        decoder = glyphs.Decoder(fonts)
        text = decoder.decode("ofciel", "fonts/b.ttf")
        assert "" not in text and glyphs.PLACEHOLDER.search(text)
        resolved = decoder.resolve([text, decoder.decode("nal", "fonts/b.ttf")])
        assert glyphs.apply(text, resolved) == "officiel"

    def test_un_meme_code_differe_d_une_police_a_l_autre(self, fonts):
        decoder = glyphs.Decoder(fonts)
        a = decoder.decode("", "fonts/b.ttf")
        b = decoder.decode("", "fonts/c.ttf")
        assert a != b

    def test_le_vote_compte_les_mots_et_non_les_occurrences(self):
        """« Sto?k » soixante fois, lisible « stock », ne vaut pas les mots à « ï »."""
        decoder = glyphs.Decoder({})
        ph = decoder.placeholder("x", 0xE0AF)
        texts = [f"Sto{ph}k"] * 60 + [f"héro{ph}que", f"na{ph}f", f"ma{ph}s", f"égo{ph}ste"]
        assert decoder.resolve(texts)[ph] == "ï"

    def test_a_egalite_la_ligature_l_emporte(self):
        decoder = glyphs.Decoder({})
        ph = decoder.placeholder("x", 1)
        # « souffle » contre « soude », « soupe », « soute » : une seule ligature.
        assert decoder.resolve([f"sou{ph}e"])[ph] == "ffl"

    def test_sans_aucune_voix_le_glyphe_reste_marque(self):
        decoder = glyphs.Decoder({})
        ph = decoder.placeholder("x", 2)
        resolved = decoder.resolve([f"PWISH{ph}TU"])
        assert glyphs.apply(f"PWISH{ph}TU", resolved) == "PWISH�TU"

    def test_le_vocabulaire_du_livre_vote_aussi(self):
        decoder = glyphs.Decoder({})
        ph = decoder.placeholder("x", 3)
        texts = ["Les Hooligans Hirsutes, dit-il. Les Hooligans !", f"H{ph}ligans"]
        assert decoder.resolve(texts)[ph] == "oo"


# --- pages ----------------------------------------------------------------------

PAGE_H = 1400
STYLE = """
.x0{left:100px}.x1{left:130px}.x2{left:400px}.x3{left:600px}
.h0{height:50px}.h1{height:90px}.h2{height:40px}.h3{height:110px}
@font-face { font-family: ff1; src: url(fonts/a.ttf) format("truetype") }
@font-face { font-family: ff2; src: url(fonts/b.ttf) format("truetype") }
@font-face { font-family: ff3; src: url(fonts/deco.ttf) format("truetype") }
"""


def page(blocks: list[tuple[str, int, str, str]], folio: str = "", running: str = "") -> str:
    """Une page pdf2htmlEX : `blocks` = (classe x, y, classe h, html du bloc)."""
    ys = sorted({y for _, y, _, _ in blocks} | {60, PAGE_H - 60})
    css = STYLE + "".join(f".y{y}{{bottom:{y}px}}" for y in ys)
    body = ""
    if running:
        body += f'<div class="t x2 y{PAGE_H - 60} h2 ff1"><span>{running}</span></div>'
    for x, y, h, html in blocks:
        body += f'<div class="t {x} y{y} {h} ff1">{html}</div>'
    if folio:
        body += f'<div class="t x2 y60 h2 ff1"><span>{folio}</span></div>'
    return (
        '<html><head><meta name="viewport" content="width=1000, height=1400"/>'
        f'<style>{css}</style></head><body><div id="page-container">'
        f'<div class="pf"><div class="pc">{body}</div></div></div></body></html>'
    )


def write_epub(path: Path, pages: list[str], toc: list[tuple[int, str]], fonts=None) -> Path:
    """Assemble un EPUB à la main : ebooklib réécrirait les pages et perdrait leur style."""
    manifest = "".join(
        f'<item id="p{i}" href="pageNum-{i}.html" media-type="application/xhtml+xml"/>'
        for i in range(1, len(pages) + 1)
    )
    manifest += "".join(
        f'<item id="f{i}" href="{href}" media-type="font/ttf"/>'
        for i, href in enumerate(fonts or {})
    )
    spine = "".join(f'<itemref idref="p{i}"/>' for i in range(1, len(pages) + 1))
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
        'unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<dc:identifier id="id">essai</dc:identifier><dc:title>Le registre</dc:title>'
        "<dc:creator>Anonyme</dc:creator><dc:language>fr</dc:language></metadata>"
        f'<manifest><item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        f'{manifest}</manifest><spine toc="ncx">{spine}</spine></package>'
    )
    points = "".join(
        f'<navPoint id="n{i}" playOrder="{i}"><navLabel><text>{title}</text></navLabel>'
        f'<content src="pageNum-{p}.html"/></navPoint>'
        for i, (p, title) in enumerate(toc, start=1)
    )
    # Comme les vrais : après les chapitres, une entrée par page, nommée de son numéro.
    points += "".join(
        f'<navPoint id="pg{i}" playOrder="{100 + i}"><navLabel><text>{i}</text></navLabel>'
        f'<content src="pageNum-{i}.html"/></navPoint>'
        for i in range(1, len(pages) + 1)
    )
    ncx = (
        '<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        f"<head/><docTitle><text>Le registre</text></docTitle><navMap>{points}</navMap></ncx>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" '
            'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            '<rootfile full-path="OPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>",
        )
        zf.writestr("OPS/content.opf", opf)
        zf.writestr("OPS/toc.ncx", ncx)
        for i, html in enumerate(pages, start=1):
            zf.writestr(f"OPS/pageNum-{i}.html", html)
        for href, data in (fonts or {}).items():
            zf.writestr(f"OPS/{href}", data)
    return path


@pytest.fixture
def book(tmp_path, fonts):
    """Trois pages liminaires, deux chapitres, une page d'illustration, des folios."""
    deco = build_font({ord("A"): "A"}, {"A": PENTAGON})
    front = page([("x0", 1200, "h0", "<span>Tous droits réservés.</span>")])
    chapter_one = page(
        [
            ("x0", 1300, "h1", "<span>1. Le départ</span>"),
            # Fragments recollés sans espace parasite, tiret de coupure en fin de ligne.
            (
                "x1",
                1200,
                "h0",
                '<span>Il</span><span class="_ _0"/><span> par</span><span class="_ _1"/>'
                '<span>tit</span><span class="_ _2"> </span><span>à l’au-</span>',
            ),
            ("x0", 1140, "h0", "<span>be, sans un mot pour les siens.</span>"),
            ("x1", 1080, "h0", "<span>— Adieu, dit-il.</span>"),
            # Lettrine : la grande lettre, puis deux lignes qui la bordent.
            ("x0", 900, "h3", "<span>L</span>"),
            ("x1", 960, "h0", "<span>orsque la nuit tomba, le vent</span>"),
            ("x1", 900, "h0", "<span>se leva sur la lande déserte.</span>"),
            # Une étiquette de dessin dans la police de titrage.
            ("x3", 700, "h0", '<span class="ff3">Adieu !</span>'),
        ],
        folio="9",
        running="LE REGISTRE",
    )
    chapter_one_bis = page(
        [
            ("x1", 1300, "h0", "<span>Il marcha toute la nuit et ne s’arrêta</span>"),
            ("x0", 1240, "h0", "<span>qu’au matin.</span>"),
        ],
        folio="10",
        running="LE REGISTRE",
    )
    illustration = page([("x2", 700, "h1", '<span class="ff3">A</span>')])
    chapter_two = page(
        [
            ("x0", 1300, "h1", "<span>2. La traversée</span>"),
            # Glyphes privés : « e » et « n » par les contours, « fi » par le dictionnaire.
            (
                "x1",
                1200,
                "h0",
                '<span>La mer était </span><span class="ff2">\ue048\ue051</span>'
                '<span> colère et l’of</span><span class="ff2">\ue0df</span>'
                "<span>cier se tut,</span>",
            ),
            ("x0", 1140, "h0", "<span>puis la houle retomba.</span>"),
            (
                "x1",
                1080,
                "h0",
                '<span>En</span><span class="ff2">\ue0df</span><span>n il parla.</span>',
            ),
        ],
        folio="11",
        running="LE REGISTRE",
    )
    pages = [front, front, front, chapter_one, chapter_one_bis, illustration, chapter_two]
    toc = [(4, "1. Le départ"), (7, "2. La traversée")]
    return write_epub(tmp_path / "fixe.epub", pages, toc, {**fonts, "fonts/deco.ttf": deco})


class TestLecture:
    def test_reconnu_comme_mise_en_page_fixe(self, book):
        doc = E.ingest(book)
        assert doc.notes["layout"].startswith("pages fixes")

    def test_chapitres_depuis_le_sommaire_sans_les_pages(self, book):
        doc = E.ingest(book)
        assert [c.title for c in doc.chapters] == ["Le départ", "La traversée"]
        assert doc.chapters[0].source_pages == (4, 6)

    def test_pages_liminaires_ecartees_et_tracees(self, book):
        doc = E.ingest(book)
        assert not any("droits réservés" in p for c in doc.chapters for p in c.paragraphs)
        assert any("3 pages liminaires" in note for note in doc.notes["skipped"])

    def test_fragments_recolles_et_coupure_reprise(self, book):
        doc = E.ingest(book)
        first = doc.chapters[0].paragraphs
        assert first[0] == "Il partit à l’aube, sans un mot pour les siens."

    def test_dialogue_et_retrait_ouvrent_un_paragraphe(self, book):
        doc = E.ingest(book)
        assert doc.chapters[0].paragraphs[1] == "— Adieu, dit-il."

    def test_lettrine_recollee(self, book):
        doc = E.ingest(book)
        assert doc.chapters[0].paragraphs[2] == (
            "Lorsque la nuit tomba, le vent se leva sur la lande déserte."
        )

    def test_le_texte_continue_d_une_page_a_l_autre(self, book):
        doc = E.ingest(book)
        assert (
            doc.chapters[0].paragraphs[3] == "Il marcha toute la nuit et ne s’arrêta qu’au matin."
        )

    def test_folios_et_titres_courants_ecartes(self, book):
        doc = E.ingest(book)
        text = "\n".join(p for c in doc.chapters for p in c.paragraphs)
        assert "LE REGISTRE" not in text
        assert not any(p in {"9", "10", "11"} for c in doc.chapters for p in c.paragraphs)

    def test_titre_du_chapitre_absent_du_corps(self, book):
        doc = E.ingest(book)
        assert not any("Le départ" in p for p in doc.chapters[0].paragraphs)

    def test_etiquettes_et_page_d_illustration_ecartees(self, book):
        doc = E.ingest(book)
        text = "\n".join(p for c in doc.chapters for p in c.paragraphs)
        assert "Adieu !" not in text
        assert any("étiquette" in note for note in doc.notes["skipped"])
        assert any("sans texte" in note for note in doc.notes["skipped"])

    def test_glyphes_prives_decodes(self, book):
        doc = E.ingest(book)
        assert doc.chapters[1].paragraphs == [
            "La mer était en colère et l’officier se tut, puis la houle retomba.",
            "Enfin il parla.",
        ]
        assert doc.needs_review is False
        assert "décodés par les contours" in doc.notes["glyphes"]


class TestRegles:
    def test_une_lettre_seule_n_est_pas_un_folio_romain(self):
        pages = [
            F.Page("p", [F.Line("L", 100, 900, 110, 0), F.Line("Il vint.", 130, 900, 50, 0)], 1400)
        ]
        F.classify(pages, 50.0)
        assert [line.running for line in pages[0].lines] == [False, False]

    def test_lettrage_eparpille_ecarte(self):
        assert F._lettering("K r o k … m o u")
        assert not F._lettering("— Oh ! Ah !")

    def test_le_titre_courant_est_reconnu_a_sa_repetition(self):
        pages = [
            F.Page("p", [F.Line("Registre du gardien", 100, 1350, 50, i)], 1400) for i in range(3)
        ]
        F.classify(pages, 50.0)
        assert all(page.lines[0].running for page in pages)

    def test_un_texte_contournant_une_image_reste_un_paragraphe(self):
        lines = [
            F.Line("Un géant de deux", 100, 1300, 50, 0),
            F.Line("mètres dix au regard", 400, 1240, 50, 0),
            F.Line("dément, dont la barbe", 400, 1180, 50, 0),
            F.Line("Halen pouffa.", 130, 1120, 50, 0),
        ]
        assert F.paragraphs([F.Page("p", lines, 1400)], 50.0) == [
            "Un géant de deux mètres dix au regard dément, dont la barbe",
            "Halen pouffa.",
        ]

    def test_coupure_en_capitales(self):
        lines = [F.Line("— FORMI-", 130, 1300, 50, 0), F.Line("DABLE !", 100, 1240, 50, 0)]
        assert F.paragraphs([F.Page("p", lines, 1400)], 50.0) == ["— FORMIDABLE !"]

    def test_un_tiret_isole_rejoint_sa_ligne(self):
        page = F.Page("p", [F.Line("très", 100, 1300, 50, 0), F.Line("-", 700, 1290, 50, 0)], 1400)
        F.merge_baselines(page, set())
        assert [line.text for line in page.lines] == ["très-"]

    def test_deux_fragments_d_un_mot_se_recollent_par_le_dictionnaire(self):
        assert F._joiner("des rochers inconfor", "tables", "fr") == ""
        assert F._joiner("des rochers", "durs", "fr") == " "

    def test_un_sommaire_absent_se_remplace_par_les_titres(self):
        toc = {"pageNum-1.html": "1", "pageNum-2.html": "2"}
        assert F.chapter_starts(["pageNum-1.html", "pageNum-2.html"], toc) == []


class TestTitre:
    def test_le_rang_du_livre_est_retire(self):
        assert clean_title("4. Comment dresser votre dragon") == "Comment dresser votre dragon"
        assert clean_title("12 – Le retour") == "Le retour"
        assert clean_title("1984") == "1984"
        assert clean_title("20 000 lieues") == "20 000 lieues"
