"""Signale les mots suspects d'un texte océrisé, avec leur contexte, pour la relecture.

Ce n'est pas un correcteur automatique, et ce serait une erreur d'en faire un : l'OCR
produit des fautes qui ne se devinent que d'après le sens — un « 1l » mis pour « il » est
une correction évidente pour un humain, indécidable pour une machine. Le rôle de ce
module est de réduire des dizaines de milliers de mots à la centaine de formes qui
méritent un coup d'œil.

Les faux positifs courants sont écartés :
  - les inversions sujet-verbe (« hurla-t-elle », « par-delà ») dont chaque partie est un
    mot connu ;
  - les noms propres, reconnus à leur capitale et à leur répétition dans le texte — un
    dictionnaire général ignore par construction les personnages d'un roman.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from spellchecker import SpellChecker

WORD_RE = re.compile(r"[A-Za-zÀ-ÿœŒæÆ'’-]{2,}")
CONTEXT = 45
PROPER_NOUN_MIN_HITS = 2

# Formes élidées : le dictionnaire ne connaît que le mot plein.
ELISIONS = {"l", "d", "j", "n", "m", "t", "s", "c", "qu", "jusqu", "lorsqu", "puisqu", "quelqu"}

# Clés de l'en-tête YAML des chapitres, qui ne sont pas du texte à relire.
STRUCTURAL = {"chapter", "title", "pages"}

# Motifs typiques de l'OCR, bien plus spécifiques que l'absence au dictionnaire :
# casse qui change au milieu d'un mot, chiffre collé à une lettre, ligature parasite.
OCR_PATTERNS = [
    (re.compile(r"[a-zà-ÿ][A-ZÀ-Ý]"), "casse interne"),
    (re.compile(r"[A-ZÀ-Ý]{2,}[a-zà-ÿ]"), "capitales suivies de minuscules"),
    (re.compile(r"\d[A-Za-zÀ-ÿ]|[A-Za-zÀ-ÿ]\d"), "chiffre collé"),
    (re.compile(r"ï[aeiouy]|[aeiouy]ï[a-z]"), "tréma suspect"),
]


def candidates(word: str) -> list[str]:
    """Décompose un mot en formes à vérifier (élisions et traits d'union).

    La ligature « œ » est développée : le dictionnaire stocke « oeuf », pas « œuf ».
    """
    core = word.strip("'’-").lower().replace("’", "'")
    core = core.replace("œ", "oe").replace("æ", "ae")
    parts = [p for p in re.split(r"['-]", core) if p and p not in ELISIONS]
    return parts if parts else [core]


@dataclass(frozen=True)
class Suspect:
    """Une forme à vérifier, située dans son texte."""

    word: str
    reason: str
    context: str
    chapter: str
    offset: int


def inspect_texts(
    texts: dict[str, str],
    language: str = "fr",
    vocabulary: Iterable[str] = (),
) -> list[Suspect]:
    """Relève les formes suspectes de plusieurs chapitres.

    `vocabulary` est le lexique propre à l'ouvrage : néologismes, noms de lieux, termes
    techniques. Il est fourni par l'utilisateur au fil de sa relecture plutôt que codé
    dans l'outil — le vocabulaire d'un livre n'a aucun sens pour le suivant.
    """
    spell = SpellChecker(language=language)
    allowed = STRUCTURAL | {w.lower() for w in vocabulary}
    corpus = "\n".join(texts.values())

    # Un mot capitalisé qui revient plusieurs fois est un nom propre, pas une coquille.
    counts = Counter(WORD_RE.findall(corpus))
    proper_nouns = {
        w.lower() for w, n in counts.items() if w[:1].isupper() and n >= PROPER_NOUN_MIN_HITS
    }

    suspects: list[Suspect] = []
    for name, text in texts.items():
        for match in WORD_RE.finditer(text):
            word = match.group()
            reason = next((label for pattern, label in OCR_PATTERNS if pattern.search(word)), None)
            if reason is None:
                if word.lower() in proper_nouns or word.lower() in allowed:
                    continue
                forms = candidates(word)
                if all(form in spell or form in allowed or len(form) < 2 for form in forms):
                    continue
                reason = "hors dictionnaire"
            start, end = match.span()
            suspects.append(
                Suspect(
                    word=word,
                    reason=reason,
                    context=text[max(0, start - CONTEXT) : end + CONTEXT].replace("\n", " "),
                    chapter=name,
                    offset=start,
                )
            )
    return suspects


def inspect_directory(directory: Path, **kwargs) -> list[Suspect]:
    # L'en-tête YAML n'est pas du texte à lire : il n'a pas à être vérifié.
    texts = {
        path.name: path.read_text(encoding="utf-8").split("---", 2)[-1]
        for path in sorted(directory.glob("ch*.md"))
    }
    return inspect_texts(texts, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--vocabulary", type=Path, help="fichier de mots propres à l'ouvrage")
    args = parser.parse_args()

    words = args.vocabulary.read_text(encoding="utf-8").split() if args.vocabulary else []
    suspects = inspect_directory(args.directory, language=args.language, vocabulary=words)

    current = ""
    for suspect in suspects:
        if suspect.chapter != current:
            current = suspect.chapter
            print(f"\n=== {current} ===")
        print(f"  {suspect.word:<18} [{suspect.reason:<26}] …{suspect.context}…")
    print(f"\n{len(suspects)} formes suspectes au total.")


if __name__ == "__main__":
    main()
