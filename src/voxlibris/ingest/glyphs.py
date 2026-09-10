"""Décodage des glyphes privés d'un EPUB issu de pdf2htmlEX.

Un PDF converti en EPUB par pdf2htmlEX conserve ses polices, et le texte reste du texte
tant que le PDF savait dire quel caractère se cachait derrière chaque glyphe. Quand il
ne le savait pas — c'est le cas des instances de police porteuses de ligatures, « fi »,
« fl », dont la table ToUnicode est souvent fausse — le convertisseur renonce et code
chaque glyphe dans la zone à usage privé d'Unicode : « officiel » devient une suite de
points de code U+E0xx, illisible pour un moteur de synthèse comme pour un lecteur.

On récupère le texte en deux temps.

1. **Par les contours.** Le livre embarque en général la même police deux fois : une
   instance codée en Unicode et l'instance fautive. Un glyphe privé dont le dessin est
   identique, au point près, à celui d'un glyphe Unicode d'une police jumelle est ce
   caractère-là. Cela décode tout ce qui existe dans les deux instances : lettres,
   accents, ponctuation.
2. **Par le dictionnaire.** Ce qui reste, ce sont les ligatures, qui n'ont de dessin
   que dans l'instance fautive. Mais un même code désigne toujours la même ligature, et
   un mot n'en tolère qu'une : « of?ciel » n'a qu'une lecture. On essaie chaque
   candidat dans chaque mot où le code apparaît, et le dictionnaire vote.

Un code qui n'obtient aucune voix reste marqué d'un « � », et le document est signalé
à relire : mieux vaut un mot visiblement cassé qu'un mot inventé.
"""

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict

PUA = range(0xE000, 0xF900)
# Les codes non décodés sont remplacés provisoirement par des caractères d'un plan privé
# supplémentaire, que les polices d'un livre n'utilisent jamais : un par couple
# (police, code), pour qu'un même code de deux polices différentes ne se confonde pas.
PLACEHOLDER_BASE = 0xF0000
UNRESOLVED = "�"

# Candidats pour un glyphe inconnu au sein d'un mot : les ligatures d'abord, ce sont
# elles qui échappent le plus souvent au décodage par contours, puis les lettres seules,
# puis toute paire de lettres, car certaines polices de titrage lient « tt » ou « nn ».
LIGATURES = ("fi", "fl", "ff", "ffi", "ffl", "ft", "st", "ct", "tt", "nn")
LETTERS = "abcdefghijklmnopqrstuvwxyzàâäçéèêëîïôöùûüÿœæ"
PAIRS = tuple(a + b for a in LETTERS[:26] for b in LETTERS[:26])
CANDIDATES = LIGATURES + tuple(LETTERS) + tuple(p for p in PAIRS if p not in LIGATURES)

# Un mot : lettres, ou substituts en attente de décodage.
WORD = re.compile(r"(?:[^\W\d_]|[\U000F0000-\U000FFFFD])+")
PLACEHOLDER = re.compile(r"[\U000F0000-\U000FFFFD]")


def _signature(glyph_set, name: str) -> tuple | None:
    """Décrit le dessin d'un glyphe, aux arrondis près, pour le comparer à un autre."""
    from fontTools.pens.recordingPen import RecordingPen

    pen = RecordingPen()
    try:
        glyph_set[name].draw(pen)
    except Exception:
        return None
    return tuple(
        (op, tuple(tuple(round(v) for v in point) for point in points)) for op, points in pen.value
    )


class Decoder:
    """Traduit les caractères privés d'un livre, police par police."""

    def __init__(self, fonts: dict[str, bytes]):
        self.fonts = fonts
        self._tables: dict[str, dict[int, str]] | None = None
        self._placeholders: dict[tuple[str, int], str] = {}
        self.available = True

    # -- contours ------------------------------------------------------------------

    def _build(self) -> dict[str, dict[int, str]]:
        try:
            from fontTools.ttLib import TTFont
        except ImportError:
            self.available = False
            return {}

        known: dict[tuple, str] = {}
        private: dict[str, dict[int, tuple | None]] = {}
        for href, data in self.fonts.items():
            try:
                font = TTFont(io.BytesIO(data))
                cmap = font.getBestCmap() or {}
                glyphs = font.getGlyphSet()
                scale = font["head"].unitsPerEm
            except Exception:
                continue
            for code, name in cmap.items():
                signature = _signature(glyphs, name)
                if signature is None:
                    continue
                key = (scale, signature)
                if code in PUA:
                    private.setdefault(href, {})[code] = key
                else:
                    known.setdefault(key, chr(code))

        tables: dict[str, dict[int, str]] = {}
        for href, codes in private.items():
            table: dict[int, str] = {}
            for code, key in codes.items():
                # Un glyphe sans dessin est une espace, quelle que soit la première
                # espace rencontrée — insécable, fine — qui lui ressemblerait.
                if not key[1]:
                    table[code] = " "
                elif key in known:
                    table[code] = known[key]
            tables[href] = table
        return tables

    @property
    def tables(self) -> dict[str, dict[int, str]]:
        if self._tables is None:
            self._tables = self._build()
        return self._tables

    def placeholder(self, href: str, code: int) -> str:
        key = (href, code)
        if key not in self._placeholders:
            self._placeholders[key] = chr(PLACEHOLDER_BASE + len(self._placeholders))
        return self._placeholders[key]

    def decode(self, text: str, href: str) -> str:
        """Remplace les caractères privés de `text`, composé dans la police `href`."""
        if not any(ord(ch) in PUA for ch in text):
            return text
        table = self.tables.get(href, {})
        out = []
        for ch in text:
            code = ord(ch)
            if code not in PUA:
                out.append(ch)
            elif code in table:
                out.append(table[code])
            else:
                out.append(self.placeholder(href, code))
        return "".join(out)

    # -- dictionnaire --------------------------------------------------------------

    def resolve(self, texts: list[str], language: str = "fr") -> dict[str, str]:
        """Fait voter le dictionnaire pour chaque substitut encore présent dans `texts`.

        Renvoie le remplacement retenu par substitut ; les codes sans voix — ou dont
        plusieurs lectures se valent sur une unique occurrence — reçoivent `UNRESOLVED`.
        """
        pending = {ph for text in texts for ph in PLACEHOLDER.findall(text)}
        if not pending:
            return {}
        spell = dictionary(language)
        # Le livre est son propre dictionnaire : un nom propre lu cent fois sans
        # ligature — « Hooligans » — tranche pour l'occurrence qui en porte une.
        vocabulary = {
            match.group().lower()
            for text in texts
            for match in WORD.finditer(text)
            if not PLACEHOLDER.search(match.group())
        }
        # On compte les mots distincts, pas les occurrences : un nom propre répété
        # soixante fois — « Sto?k », que « stock » lirait sans peine — ne doit pas
        # l'emporter sur les sept mots ordinaires qui, eux, veulent un « ï ».
        words = {
            match.group()
            for text in texts
            for match in WORD.finditer(text)
            if len(PLACEHOLDER.findall(match.group())) == 1
        }
        votes: dict[str, Counter[str]] = defaultdict(Counter)
        for word in words:
            ph = PLACEHOLDER.search(word).group()
            for candidate in CANDIDATES:
                form = word.replace(ph, candidate).lower()
                if len(form) >= 3 and (form in vocabulary or spell.known([form])):
                    votes[ph][candidate] += 1

        resolved: dict[str, str] = {}
        for ph in pending:
            tally = votes.get(ph)
            if not tally:
                resolved[ph] = UNRESOLVED
                continue
            best = max(tally.values())
            tied = [c for c, n in tally.items() if n == best]
            # À égalité, une ligature l'emporte sur une lettre quelconque : c'est elle
            # que le convertisseur perd, pas le « d » de « soude ».
            ligatures = [c for c in tied if c in LIGATURES]
            if len(tied) == 1:
                resolved[ph] = tied[0]
            elif len(ligatures) == 1:
                resolved[ph] = ligatures[0]
            else:
                resolved[ph] = UNRESOLVED
        return resolved

    def describe(self, resolved: dict[str, str]) -> dict[str, object]:
        """Résumé pour les notes du document."""
        by_font: Counter[str] = Counter()
        for href, _code in self._placeholders:
            by_font[href] += 1
        decoded = sum(len(t) for t in self.tables.values())
        unresolved = sum(1 for v in resolved.values() if v == UNRESOLVED)
        return {
            "decoded": decoded,
            "voted": sum(1 for v in resolved.values() if v != UNRESOLVED),
            "unresolved": unresolved,
        }


_dictionaries: dict[str, object] = {}


def dictionary(language: str = "fr"):
    """Dictionnaire de la langue, chargé une fois — il pèse une seconde."""
    from spellchecker import SpellChecker

    if language not in _dictionaries:
        _dictionaries[language] = SpellChecker(language=language)
    return _dictionaries[language]


def apply(text: str, resolved: dict[str, str]) -> str:
    """Remplace les substituts d'un texte par la lecture retenue."""
    if not resolved or not PLACEHOLDER.search(text):
        return text
    return PLACEHOLDER.sub(lambda m: resolved.get(m.group(), UNRESOLVED), text)
