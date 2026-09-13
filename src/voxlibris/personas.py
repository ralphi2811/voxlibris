"""Repérage des personas par un modèle de langage.

Le modèle ne décide de rien : il propose. Pour chaque chapitre, il reçoit les paragraphes
numérotés et répond par les personas qu'il y voit et les blocs qui leur appartiennent —
une lettre, un journal, un récit enchâssé, un poème : ce qu'un personnage écrit ou dit
d'un seul tenant, sur plusieurs paragraphes. Pas les répliques : une réplique isolée est
l'affaire de la règle typographique, et couper les incises est un autre travail.

Chaque proposition passe ensuite des garde-fous mécaniques : les bornes doivent exister,
un bloc ne chevauche pas l'autre, le persona est dans la liste annoncée. Ce qui reste
arrive dans la Relecture, où un humain attribue ou ignore ; attribuer pose les marqueurs
« @Nom » et « @ » que la préparation lit. Rien n'est écrit sans ce geste.

Le texte entier part au modèle, chapitre par chapitre. Local par défaut, comme le reste
de l'assistance ; l'avertissement sur un service distant s'applique de la même façon.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field

from .llm import LLM, LLMConfig, LLMError
from .normalize import NARRATOR, persona_of

SYSTEM = """\
Tu lis un chapitre de livre en français, paragraphe par paragraphe, pour repérer les \
passages qui ne sont pas dits par le narrateur mais par un personnage, d'un seul tenant, \
sur un ou plusieurs paragraphes : une lettre, un journal intime, un récit qu'un \
personnage raconte, un poème, une chanson, un article lu.

Tu ne t'occupes pas des dialogues : une réplique isolée, même longue, n'est pas un bloc. \
Une lettre commence à sa salutation et finit à sa signature ; le récit qui l'introduit \
et celui qui la commente restent au narrateur.

Tu réponds uniquement par un objet JSON :
{"personas": [{"nom": "Charles", "qui": "le frère de Lulu, soldat"}],
 "blocs": [{"persona": "Charles", "de": 12, "a": 20}]}

« de » et « a » sont les numéros du premier et du dernier paragraphe du bloc, tels qu'ils \
te sont donnés. Un persona sans bloc ne sert à rien : ne le cite pas. Quand un persona \
t'est annoncé comme déjà connu, reprends son nom à l'identique. S'il n'y a aucun bloc, \
réponds {"personas": [], "blocs": []}.
"""

# Au-delà, un modèle local perd le fil, et sa fenêtre de contexte aussi. Un chapitre
# plus long est présenté en tranches ; un bloc à cheval sur deux tranches est perdu, ce
# qui est dit dans le rapport.
SLICE_CHARS = 24_000


@dataclass(frozen=True)
class Persona:
    name: str
    who: str = ""


@dataclass(frozen=True)
class Block:
    """Une plage de paragraphes qu'un persona dirait, bornée par le texte de ses deux
    paragraphes extrêmes — c'est ce qui la retrouve dans l'éditeur, même si le texte a
    bougé autour."""

    chapter: str
    persona: str
    first: int
    last: int
    opening: str
    closing: str


@dataclass
class Report:
    personas: list[Persona] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    chapters: int = 0
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "personas": [asdict(p) for p in self.personas],
                "blocks": [asdict(b) for b in self.blocks],
                "rejected": self.rejected,
                "chapters": self.chapters,
                "error": self.error,
            },
            ensure_ascii=False,
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str) -> Report:
        data = json.loads(text)
        return cls(
            personas=[Persona(**p) for p in data.get("personas", [])],
            blocks=[Block(**b) for b in data.get("blocks", [])],
            rejected=data.get("rejected", []),
            chapters=data.get("chapters", 0),
            error=data.get("error", ""),
        )


def paragraphs_of(text: str) -> list[str]:
    """Les paragraphes tels que l'éditeur les voit : séparés par une ligne vide."""
    return [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]


def readers(paragraphs: Iterable[str]) -> list[str]:
    """Qui lit chaque paragraphe d'après les marqueurs déjà posés ; vide pour un
    marqueur lui-même. Sert à ne pas reproposer ce qui est fait."""
    out, reader = [], ""
    for paragraph in paragraphs:
        marker = persona_of(paragraph)
        if marker is not None:
            reader = marker
            out.append("")
        else:
            out.append(reader or NARRATOR)
    return out


def _slices(paragraphs: list[str], limit: int) -> Iterable[tuple[int, list[str]]]:
    start, size = 0, 0
    for index, paragraph in enumerate(paragraphs):
        if size and size + len(paragraph) > limit:
            yield start, paragraphs[start:index]
            start, size = index, 0
        size += len(paragraph)
    if start < len(paragraphs):
        yield start, paragraphs[start:]


def _ask(llm: LLM, numbered: list[tuple[int, str]], known: list[str]) -> dict:
    lines = "\n".join(f"[{n}] {p}" for n, p in numbered)
    prompt = (
        (f"Personas déjà connus : {', '.join(known)}.\n\n" if known else "")
        + "Paragraphes du chapitre :\n\n"
        + lines
    )
    answer = llm.ask_json(prompt, system=SYSTEM)
    if not isinstance(answer, dict):
        raise LLMError("Le modèle n'a pas renvoyé d'objet")
    return answer


def _same(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def discover(
    texts: dict[str, str],
    known: Iterable[str] = (),
    config: LLMConfig | None = None,
    llm: LLM | None = None,
    on_chapter: Callable[[str, int, int], None] = lambda *_: None,
) -> Report:
    """Demande au modèle les personas et leurs blocs, chapitre par chapitre.

    Les noms retenus au fil des chapitres sont annoncés aux suivants, pour qu'un même
    personnage garde un seul nom. Une panne du modèle n'interrompt rien : le rapport
    revient avec ce qui a été obtenu, et son motif.
    """
    config = config or (llm.config if llm else LLMConfig.from_env())
    report = Report()
    if not config.enabled:
        report.error = "Assistance désactivée (VOXLIBRIS_LLM_ENABLED=0)"
        return report

    names: list[str] = [n for n in known if n]
    llm = llm or LLM(config)
    items = sorted(texts.items())
    for position, (chapter, text) in enumerate(items):
        on_chapter(chapter, position, len(items))
        report.chapters += 1
        paragraphs = paragraphs_of(text)
        # Les marqueurs déjà posés ne sont pas des paragraphes à lire.
        readable = [(i, p) for i, p in enumerate(paragraphs) if persona_of(p) is None]
        current = readers(paragraphs)
        taken: list[tuple[int, int]] = []
        for start, chunk in _slices([p for _, p in readable], SLICE_CHARS):
            numbered = [(readable[start + k][0], p) for k, p in enumerate(chunk)]
            try:
                answer = _ask(llm, numbered, names)
            except LLMError as error:
                report.error = str(error)
                return report
            valid = {n for n, _ in numbered}
            for entry in answer.get("personas") or []:
                if not isinstance(entry, dict) or not str(entry.get("nom") or "").strip():
                    continue
                name = str(entry["nom"]).strip()
                if _same(name, NARRATOR):
                    continue
                if not any(_same(name, n) for n in names):
                    names.append(name)
                    report.personas.append(Persona(name, str(entry.get("qui") or "").strip()))
            for entry in answer.get("blocs") or []:
                if not isinstance(entry, dict):
                    continue
                persona = next(
                    (n for n in names if _same(n, str(entry.get("persona") or ""))), None
                )
                first, last = entry.get("de"), entry.get("a")
                if persona is None:
                    report.rejected.append(f"{chapter} : persona inconnu {entry.get('persona')!r}")
                    continue
                if not (isinstance(first, int) and isinstance(last, int)) or first > last:
                    report.rejected.append(f"{chapter} : bornes illisibles {first!r}–{last!r}")
                    continue
                if first not in valid or last not in valid:
                    report.rejected.append(f"{chapter} : bornes hors du chapitre {first}–{last}")
                    continue
                if any(a <= last and first <= b for a, b in taken):
                    report.rejected.append(
                        f"{chapter} : {persona} {first}–{last} en chevauche un autre"
                    )
                    continue
                if all(_same(current[i], persona) for i in range(first, last + 1)):
                    continue  # déjà attribué dans le texte : rien à proposer
                taken.append((first, last))
                report.blocks.append(
                    Block(chapter, persona, first, last, paragraphs[first], paragraphs[last])
                )
    return report


def pending(blocks: Iterable[Block], text: str) -> list[tuple[Block, int]]:
    """Les blocs qui restent à attribuer dans ce texte, chacun avec le rang de son
    paragraphe d'ouverture parmi ses homonymes.

    Leurs bornes doivent s'y trouver encore, et le passage ne pas être déjà lu par ce
    persona. Deux lettres du même frère ouvrent sur le même « Ma Lulu, » : on vise
    l'occurrence la plus proche du numéro relevé, et l'éditeur reçoit ce rang pour
    retrouver la même.
    """
    paragraphs = paragraphs_of(text)
    current = readers(paragraphs)
    out = []
    for block in blocks:
        openings = [i for i, p in enumerate(paragraphs) if p == block.opening]
        if not openings:
            continue
        first = min(openings, key=lambda i: abs(i - block.first))
        try:
            last = paragraphs.index(block.closing, first)
        except ValueError:
            continue
        if all(_same(current[i], block.persona) for i in range(first, last + 1)):
            continue
        out.append((block, openings.index(first)))
    return out
