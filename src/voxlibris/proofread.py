"""Suggestions de correction assistées par un modèle de langage.

Le modèle ne réécrit jamais le texte. Il ne voit qu'une chose à la fois : une forme déjà
signalée par `review.py`, dans son contexte, et il ne peut proposer qu'un remplacement
pour cette forme-là. Tout le reste du texte lui est inaccessible par construction.

Ce cadrage n'est pas de la prudence de principe, il répond à un danger précis. Un
modèle à qui l'on demande de « relire et améliorer » corrige ce qui n'est pas fautif :
le parler d'un personnage, une graphie d'époque, une ponctuation calibrée pour la
synthèse vocale. Rien ne le signale, et la perte est irréversible faute de savoir où
regarder. En n'exposant que les formes suspectes, on borne le pire cas.

Trois garde-fous filtrent ensuite chaque proposition :

  - la distance d'édition est bornée — une coquille d'OCR est une lettre confondue, pas
    un mot réécrit ;
  - le remplacement doit exister au dictionnaire, ou déjà figurer dans l'ouvrage, ce qui
    élimine les inventions ;
  - aucune ponctuation ne peut être introduite, la ponctuation étant réglée pour le TTS.

Et rien n'est appliqué : les suggestions alimentent l'éditeur de relecture, où un humain
tranche. L'apport n'est pas de supprimer la relecture, c'est de la ramener de « lire
quatorze mille mots » à « statuer sur une centaine de propositions ».
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

from spellchecker import SpellChecker

from .llm import LLM, LLMConfig, LLMError
from .review import Suspect, candidates, inspect_texts

# Une coquille d'OCR reste proche du mot d'origine. Au-delà, ce n'est plus une lecture
# corrigée mais une réécriture, et le mot juste est indécidable sans le sens.
MAX_DISTANCE = 3

# « pardessus » → « par-dessus » : une correction peut scinder un mot, pas rédiger.
MAX_WORDS = 2

# La ponctuation a été calibrée pour la synthèse (points de suspension convertis en
# silence, deux-points conservés). Le modèle n'a pas à y toucher.
FORBIDDEN = set('.!?;:,«»"()[]{}…')

# Une forme qui revient est rarement une coquille : c'est un nom propre, un néologisme,
# ou le parler d'un personnage. Les soumettre au modèle, c'est l'inviter à normaliser ce
# que l'auteur a voulu. Au-delà de ce seuil, la forme est laissée au seul jugement humain.
RECURRENCE_LIMIT = 4

WORDS = re.compile(r"[A-Za-zÀ-ÿœŒæÆ'’-]+")

SYSTEM = """\
Tu corriges des erreurs de reconnaissance optique de caractères (OCR) dans un roman en \
français. Tu ne fais rien d'autre.

Pour chaque forme signalée, tu réponds soit par le mot que l'OCR a mal lu, soit par null.

Tu réponds null — et c'est le cas le plus fréquent — dès que :
- la forme est correcte telle quelle ;
- c'est un nom propre, un nom de lieu ou de navire ;
- c'est un parler populaire, une élision volontaire ou un archaïsme (« i' », « vot' », \
« Môssieur ») : l'auteur l'a écrit ainsi, le corriger détruirait la voix d'un personnage ;
- le contexte ne suffit pas à trancher avec certitude.

Tu ne reformules jamais, tu n'ajoutes ni ne retires aucune ponctuation, et ta réponse ne \
porte que sur la forme signalée, jamais sur le reste de la phrase.

Tu réponds uniquement par un tableau JSON, sans commentaire ni balise Markdown."""


@dataclass(frozen=True)
class Suggestion:
    """Une substitution proposée, rattachée à sa position exacte dans le texte."""

    chapter: str
    offset: int
    word: str
    replacement: str
    reason: str
    recurrence: int = 1
    # Le contexte accompagne la proposition jusqu'à l'écran. C'est lui qui permet à un
    # relecteur de rejeter d'un coup d'œil un « Mâdâme » corrigé en « Madame » : hors
    # de sa phrase, la proposition paraît raisonnable ; dedans, on entend le personnage.
    context: str = ""

    @property
    def end(self) -> int:
        return self.offset + len(self.word)


@dataclass(frozen=True)
class Rejected:
    """Une proposition écartée par les garde-fous, et le motif du rejet."""

    word: str
    proposal: str
    cause: str


@dataclass
class Report:
    suggestions: list[Suggestion] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    suspects: int = 0
    asked: int = 0
    skipped: list[str] = field(default_factory=list)
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "suggestions": [asdict(s) for s in self.suggestions],
                "rejected": [asdict(r) for r in self.rejected],
                "suspects": self.suspects,
                "asked": self.asked,
                "skipped": self.skipped,
                "error": self.error,
            },
            ensure_ascii=False,
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str) -> Report:
        data = json.loads(text)
        return cls(
            suggestions=[Suggestion(**s) for s in data.get("suggestions", [])],
            rejected=[Rejected(**r) for r in data.get("rejected", [])],
            suspects=data.get("suspects", 0),
            asked=data.get("asked", 0),
            skipped=data.get("skipped", []),
            error=data.get("error", ""),
        )


# --- Garde-fous ---------------------------------------------------------------------
def distance(a: str, b: str) -> int:
    """Distance de Levenshtein, sur des mots — quelques dizaines de caractères au plus."""
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _known(word: str, spell: SpellChecker, allowed: set[str]) -> bool:
    forms = candidates(word)
    return all(form in spell or form in allowed or len(form) < 2 for form in forms)


def check(
    word: str,
    proposal: object,
    spell: SpellChecker,
    allowed: set[str],
) -> tuple[str, str]:
    """Valide une proposition. Renvoie (remplacement, "") ou ("", motif de rejet)."""
    if proposal is None:
        return "", ""
    if not isinstance(proposal, str):
        return "", "type inattendu"

    replacement = proposal.strip()
    # Les modèles rendent volontiers l'apostrophe droite du clavier. La reprendre telle
    # quelle sèmerait deux apostrophes différentes dans un même livre.
    if "’" in word and "'" in replacement:
        replacement = replacement.replace("'", "’")
    if not replacement:
        return "", ""
    if replacement == word:
        return "", ""

    if FORBIDDEN & set(replacement):
        return "", "ponctuation introduite"
    if len(replacement.split()) > MAX_WORDS:
        return "", f"{len(replacement.split())} mots proposés"

    limit = min(MAX_DISTANCE, max(1, len(word) // 2))
    gap = distance(word.lower(), replacement.lower())
    if gap > limit:
        return "", f"distance {gap} > {limit}"

    if word[:1].isupper() and replacement[:1].islower():
        return "", "casse perdue"

    for part in replacement.split():
        if not _known(part, spell, allowed):
            return "", f"« {part} » hors dictionnaire"

    return replacement, ""


# --- Interrogation du modèle --------------------------------------------------------
def _mark(suspect: Suspect, text: str) -> str:
    """Le contexte, avec la forme signalée encadrée pour lever toute ambiguïté."""
    start, end = suspect.offset, suspect.offset + len(suspect.word)
    left = text[max(0, start - 90) : start].replace("\n", " ")
    right = text[end : end + 90].replace("\n", " ")
    return f"…{left}⟦{suspect.word}⟧{right}…"


def _batches(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _ask(llm: LLM, batch: Sequence[tuple[Suspect, str]]) -> dict[int, object]:
    entries = [
        {"id": index, "forme": suspect.word, "contexte": context}
        for index, (suspect, context) in enumerate(batch)
    ]
    prompt = (
        "Voici des formes relevées dans un texte océrisé. La forme concernée est encadrée "
        "par ⟦ ⟧ dans son contexte.\n\n"
        + json.dumps(entries, ensure_ascii=False, indent=1)
        + "\n\nRéponds par un tableau JSON de la même longueur, chaque élément de la forme "
        '{"id": <entier>, "correction": "<mot>"} ou {"id": <entier>, "correction": null}.'
    )
    answer = llm.ask_json(prompt, system=SYSTEM)
    if not isinstance(answer, list):
        raise LLMError("Le modèle n'a pas renvoyé de tableau")

    replies: dict[int, object] = {}
    for item in answer:
        if isinstance(item, dict) and isinstance(item.get("id"), int):
            replies[item["id"]] = item.get("correction")
    return replies


def suggest(
    texts: dict[str, str],
    language: str = "fr",
    vocabulary: Iterable[str] = (),
    config: LLMConfig | None = None,
    llm: LLM | None = None,
    recurrence_limit: int = RECURRENCE_LIMIT,
) -> Report:
    """Propose des corrections pour les formes suspectes de plusieurs chapitres.

    Une panne du modèle n'interrompt rien : le rapport revient vide, avec son motif. La
    relecture manuelle reste possible, l'assistance n'est qu'un confort.
    """
    config = config or (llm.config if llm else LLMConfig.from_env())
    report = Report()
    if not config.enabled:
        report.error = "Assistance désactivée (VOXLIBRIS_LLM_ENABLED=0)"
        return report

    suspects = inspect_texts(texts, language=language, vocabulary=vocabulary)
    report.suspects = len(suspects)
    if not suspects:
        return report

    forms = Counter(s.word for s in suspects)
    pending: list[tuple[Suspect, str]] = []
    for suspect in suspects:
        if forms[suspect.word] >= recurrence_limit:
            continue
        pending.append((suspect, _mark(suspect, texts[suspect.chapter])))
    report.skipped = sorted(w for w, n in forms.items() if n >= recurrence_limit)
    report.asked = len(pending)
    if not pending:
        return report

    spell = SpellChecker(language=language)
    allowed = {w.lower() for w in vocabulary}
    # Les mots déjà présents dans l'ouvrage sont un lexique légitime : un nom de
    # personnage n'est dans aucun dictionnaire, et le modèle a le droit d'y ramener.
    corpus_words = {w.lower() for w in WORDS.findall("\n".join(texts.values()))}
    allowed |= corpus_words - {s.word.lower() for s in suspects}

    llm = llm or LLM(config)
    for batch in _batches(pending, config.batch_size):
        try:
            replies = _ask(llm, batch)
        except LLMError as error:
            report.error = str(error)
            break
        for index, (suspect, context) in enumerate(batch):
            replacement, cause = check(suspect.word, replies.get(index), spell, allowed)
            if cause:
                report.rejected.append(Rejected(suspect.word, str(replies.get(index)), cause))
            elif replacement:
                report.suggestions.append(
                    Suggestion(
                        chapter=suspect.chapter,
                        offset=suspect.offset,
                        word=suspect.word,
                        replacement=replacement,
                        reason=suspect.reason,
                        recurrence=forms[suspect.word],
                        context=context,
                    )
                )
    return report


def apply(text: str, suggestions: Iterable[Suggestion]) -> str:
    """Applique des suggestions à un texte, de la fin vers le début.

    Les positions restent ainsi valides d'une substitution à l'autre. Cette fonction
    n'est appelée que sur des suggestions qu'un humain a explicitement retenues.
    """
    result = text
    for suggestion in sorted(suggestions, key=lambda s: s.offset, reverse=True):
        if result[suggestion.offset : suggestion.end] != suggestion.word:
            continue  # le texte a changé depuis : on ne corrige pas à l'aveugle
        result = result[: suggestion.offset] + suggestion.replacement + result[suggestion.end :]
    return result
