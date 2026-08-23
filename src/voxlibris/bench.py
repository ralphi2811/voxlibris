"""Banc d'essai des suggestions, contre une relecture humaine.

L'assistance par modèle de langage ne se juge pas à l'impression qu'elle donne sur
quelques exemples : elle se mesure. Il suffit pour cela d'un texte océrisé et de sa
version relue à la main, et l'on obtient trois nombres.

Le premier est flatteur — les fautes corrigées. Le deuxième est acceptable — les fautes
manquées, qui laissent simplement le travail en l'état. Le troisième est le seul qui
décide de tout : les **modifications indues**, ces propositions portant sur un texte qui
était déjà juste. Une seule acceptée par distraction, et le livre s'éloigne de l'original
sans que rien ne le signale. S'il n'est pas quasi nul, l'idée est mauvaise.

    voxlibris bench-proofread text/raw text/clean

Le diff se fait au mot, ponctuation exclue : les changements de ponctuation de la
relecture (points de suspension, tirets de dialogue) ne sont pas du ressort du modèle et
n'ont donc pas à peser sur le résultat.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .llm import LLMConfig
from .proofread import Report, Suggestion, suggest
from .review import inspect_texts

TOKEN = re.compile(r"[A-Za-zÀ-ÿœŒæÆ0-9'’-]+")


def read_bodies(directory: Path) -> dict[str, str]:
    """Le corps des chapitres, en-tête YAML retiré — comme le voit la relecture."""
    return {
        path.name: path.read_text(encoding="utf-8").split("---", 2)[-1]
        for path in sorted(directory.glob("ch*.md"))
    }


@dataclass(frozen=True)
class Correction:
    """Une différence entre le texte brut et le texte relu, située dans le brut."""

    chapter: str
    offset: int
    word: str
    expected: str


def ground_truth(raw: dict[str, str], clean: dict[str, str]) -> list[Correction]:
    """Les corrections que la relecture humaine a effectivement apportées.

    Les deux textes sont alignés d'un bloc plutôt que chapitre par chapitre : une
    relecture peut scinder un chapitre ou en ajouter un liminaire, et un alignement par
    fichier compterait alors ces déplacements comme des milliers de corrections.
    """
    raw_tokens: list[tuple[str, str, int]] = []
    for name, text in raw.items():
        raw_tokens += [(m.group(), name, m.start()) for m in TOKEN.finditer(text)]
    clean_tokens = [m.group() for text in clean.values() for m in TOKEN.finditer(text)]

    matcher = SequenceMatcher(
        None, [t[0].lower() for t in raw_tokens], [t.lower() for t in clean_tokens], autojunk=False
    )
    corrections: list[Correction] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # L'alignement ignore la casse, sans quoi une majuscule parasite décalerait
            # tout le bloc. Mais « trouVait » corrigé en « trouvait » reste une
            # correction, et la comparer ici est le seul moyen de ne pas la perdre.
            for offset in range(i2 - i1):
                word, chapter, position = raw_tokens[i1 + offset]
                if word != clean_tokens[j1 + offset]:
                    corrections.append(
                        Correction(chapter, position, word, clean_tokens[j1 + offset])
                    )
            continue
        if tag == "insert":
            continue  # du texte ajouté par la relecture : rien à corriger dans le brut
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for offset in range(i2 - i1):
                word, chapter, position = raw_tokens[i1 + offset]
                corrections.append(Correction(chapter, position, word, clean_tokens[j1 + offset]))
            continue
        # Suppression, ou remplacement de longueurs différentes : le bloc entier du brut
        # est fautif, et l'attendu est ce que la relecture a mis à la place.
        expected = " ".join(clean_tokens[j1:j2])
        for word, chapter, position in raw_tokens[i1:i2]:
            corrections.append(Correction(chapter, position, word, expected))
    return corrections


def typographic(correction: Correction) -> bool:
    """Vrai si la relecture n'a fait que redresser l'apostrophe.

    Ces différences sont nombreuses et sans intérêt ici : l'apostrophe droite est une
    convention de saisie, redressée par la normalisation typographique. Les compter
    comme des fautes manquées ferait porter au modèle un travail qui n'est pas le sien.
    """
    return correction.word.replace("'", "’") == correction.expected


@dataclass
class Outcome:
    """Le décompte d'une exécution du banc d'essai."""

    typographic: int = 0
    fixed: list[tuple[Correction, Suggestion]] = field(default_factory=list)
    inexact: list[tuple[Correction, Suggestion]] = field(default_factory=list)
    undue: list[Suggestion] = field(default_factory=list)
    missed_flagged: list[Correction] = field(default_factory=list)
    missed_unflagged: list[Correction] = field(default_factory=list)
    report: Report = field(default_factory=Report)
    corrections: int = 0

    @property
    def precision(self) -> float:
        """Part des suggestions qui portent sur un mot réellement fautif."""
        total = len(self.fixed) + len(self.inexact) + len(self.undue)
        return len(self.fixed) / total if total else 0.0

    @property
    def recall(self) -> float:
        return len(self.fixed) / self.corrections if self.corrections else 0.0


def evaluate(
    raw: dict[str, str],
    clean: dict[str, str],
    language: str = "fr",
    vocabulary: Iterable[str] = (),
    config: LLMConfig | None = None,
    **kwargs,
) -> Outcome:
    everything = ground_truth(raw, clean)
    truth = [c for c in everything if not typographic(c)]
    expected = {(c.chapter, c.offset): c for c in truth}

    report = suggest(raw, language=language, vocabulary=vocabulary, config=config, **kwargs)
    flagged = {
        (s.chapter, s.offset) for s in inspect_texts(raw, language=language, vocabulary=vocabulary)
    }

    outcome = Outcome(
        report=report,
        corrections=len(truth),
        typographic=len(everything) - len(truth),
    )
    answered: set[tuple[str, int]] = set()
    for suggestion in report.suggestions:
        key = (suggestion.chapter, suggestion.offset)
        answered.add(key)
        correction = expected.get(key)
        if correction is None:
            outcome.undue.append(suggestion)
        elif suggestion.replacement.lower() == correction.expected.lower():
            outcome.fixed.append((correction, suggestion))
        else:
            outcome.inexact.append((correction, suggestion))

    for correction in truth:
        key = (correction.chapter, correction.offset)
        if key in answered:
            continue
        target = outcome.missed_flagged if key in flagged else outcome.missed_unflagged
        target.append(correction)
    return outcome
