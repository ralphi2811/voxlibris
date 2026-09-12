"""Synthétise les segments d'un chapitre en un WAV, avec manifeste et contrôle qualité.

Chaque chapitre produit deux fichiers : `chNN.wav` et `chNN.timing.json`. Le manifeste
donne les bornes de chaque segment dans le fichier final. C'est lui qui permet de partir
d'un instant entendu pour retrouver le segment fautif — impossible autrement une fois le
chapitre concaténé, et décisif pour corriger les artefacts que seule l'oreille détecte.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backends import Backend
from .quality import SAMPLE_RATE, QualityProfile, Take, calibrate, render_with_fallback
from .voxtral import Refused

# Silence en tête de chaque piste, en millisecondes. Une piste qui démarre sur le premier
# phonème surprend l'oreille, et enchaînée à la précédente dans le livre assemblé, elle
# lui colle : les livres audio du commerce ouvrent chaque chapitre sur une demi-seconde
# de rien.
LEAD_IN_MS = 600


@dataclass
class Segment:
    idx: int
    text: str
    pause_after_ms: int
    chapter: int
    title: str


def load_segments(path: Path) -> list[Segment]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [
        Segment(
            idx=r["idx"],
            text=r["text"],
            pause_after_ms=r["pause_after_ms"],
            chapter=r["chapter"],
            title=r["title"],
        )
        for r in records
    ]


@dataclass
class ChapterResult:
    audio: np.ndarray
    timing: list[dict]
    warnings: list[str]

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    @property
    def reused(self) -> int:
        return sum(1 for entry in self.timing if entry.get("reused"))

    def write(self, target: Path) -> None:
        import soundfile as sf

        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, self.audio, SAMPLE_RATE)
        target.with_suffix(".timing.json").write_text(
            json.dumps(self.timing, ensure_ascii=False, indent=1), encoding="utf-8"
        )


def previous_takes(track: Path, timing: list[dict]) -> Callable[[str], Take | None]:
    """Les segments propres d'une piste déjà synthétisée, retrouvables par leur texte.

    Un mot corrigé dans un chapitre ne doit pas coûter le chapitre entier : tout segment
    dont le texte n'a pas bougé est repris tel quel de la piste précédente, découpé
    d'après son manifeste. Les segments signalés ne sont pas repris — c'est l'occasion
    de les rejouer. La piste n'est lue que si un segment est effectivement repris.
    """
    known = {
        str(entry["text"]): entry
        for entry in timing
        if entry.get("clean", True) and entry.get("text") and entry["end"] > entry["start"]
    }
    audio: list[np.ndarray] = []

    def lookup(text: str) -> Take | None:
        entry = known.get(text)
        if entry is None:
            return None
        if not audio:
            import soundfile as sf

            data, _ = sf.read(track, dtype="float32")
            audio.append(data[:, 0] if data.ndim > 1 else data)
        start = int(round(float(entry["start"]) * SAMPLE_RATE))
        end = int(round(float(entry["end"]) * SAMPLE_RATE))
        piece = audio[0][start:end]
        if not len(piece):
            return None
        return Take(np.array(piece, np.float32), True, int(entry.get("attempts", 1)))

    return lookup


def synthesize_chapter(
    backend: Backend,
    segments: Iterable[Segment],
    profile: QualityProfile | None = None,
    on_segment: Callable[[Segment, float, int], None] = lambda *_: None,
    reuse: Callable[[str], Take | None] | None = None,
    lead_in_ms: int = LEAD_IN_MS,
) -> ChapterResult:
    """Synthétise et concatène un chapitre, en signalant les segments douteux.

    `reuse` propose, pour un texte, une prise déjà faite : elle est alors reprise sans
    passer par le moteur — voir `previous_takes`. `lead_in_ms` est le silence en tête ;
    zéro pour un fragment destiné à être recollé dans une piste, ou un échantillon.
    """
    profile = profile or QualityProfile()
    lead_in = np.zeros(int(SAMPLE_RATE * lead_in_ms / 1000), np.float32)
    pieces: list[np.ndarray] = [lead_in]
    timing: list[dict] = []
    warnings: list[str] = []
    cursor = len(lead_in)

    for segment in segments:
        reused = reuse(segment.text) if reuse else None
        try:
            if reused is not None:
                take, split = reused, False
            else:
                take, split = render_with_fallback(backend.say, segment.text, profile)
        except Refused as refus:
            # Un moteur distant peut refuser une phrase, et il refusera les mêmes à
            # chaque tentative. Abandonner tout le livre pour autant serait absurde :
            # on laisse un silence, on le consigne, et le relecteur saura quoi reprendre.
            warnings.append(
                f"ch{segment.chapter:02d} segment {segment.idx} REFUSÉ "
                f"({', '.join(refus.categories) or 'modération'}) — « {segment.text[:80]} »"
            )
            take, split = Take(np.zeros(0, np.float32), False, 1, "refusé"), False

        if not take.clean:
            warnings.append(
                f"ch{segment.chapter:02d} segment {segment.idx} ({take.cause}) : "
                f"{take.duration:.1f}s pour {profile.expected(segment.text):.1f}s "
                f"attendues après {take.attempts} essais — « {segment.text[:60]} »"
            )

        pause = np.zeros(int(SAMPLE_RATE * segment.pause_after_ms / 1000), np.float32)
        timing.append(
            {
                "idx": segment.idx,
                "start": round(cursor / SAMPLE_RATE, 2),
                "end": round((cursor + len(take.audio)) / SAMPLE_RATE, 2),
                "attempts": take.attempts,
                "clean": take.clean,
                # La cause est ce qui permet, depuis l'interface, de trier ce qu'il faut
                # réécouter : un « trop court » n'a pas la même gravité qu'un « babil ».
                "cause": "" if take.clean else take.cause,
                "split": split,
                "text": segment.text,
                # Repris de la piste précédente, sans passer par le moteur.
                "reused": reused is not None,
            }
        )
        cursor += len(take.audio) + len(pause)
        pieces += [take.audio, pause]
        on_segment(segment, take.duration, take.attempts)

    audio = np.concatenate(pieces) if timing else np.zeros(0, np.float32)
    return ChapterResult(audio=audio, timing=timing, warnings=warnings)


def spread(items: list, count: int) -> list:
    """Prélève `count` éléments répartis sur toute la liste, plutôt que les premiers.

    Un début de chapitre n'est pas représentatif : dédicace, épigraphe, ou — cas vécu —
    la notice d'usage anglaise d'un EPUB Gutenberg. Calibrer le débit là-dessus a donné
    13,3 caractères par seconde pour une voix qui en tient 19 sur du français, et le
    contrôle qualité a ensuite rejoué 209 phrases courtes jugées « trop brèves » — trois
    rendus par segment, une demi-heure de synthèse pour un chapitre.
    """
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def profile_for_voice(
    backend: Backend,
    segments: list[Segment],
    store: Path,
    sample_size: int = 12,
) -> QualityProfile:
    """Renvoie un profil de contrôle qualité calibré sur la voix employée.

    Le débit varie sensiblement d'une voix à l'autre, et tous les seuils en dérivent :
    réutiliser tel quel le débit mesuré sur une autre voix rendrait le contrôle soit
    aveugle, soit bavard. La mesure est faite une fois par couple (moteur, voix), sur un
    échantillon de segments, puis mémorisée.
    """
    # La vitesse entre dans la clé : elle change le débit dans les mêmes proportions, et
    # réutiliser une mesure faite à un autre réglage rendrait tous les seuils faux.
    speed = getattr(backend, "speed", 1.0)
    key = f"{backend.name}/{getattr(backend, 'voice', 'default')}@{speed:.2f}"
    known = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
    if key in known:
        return QualityProfile(chars_per_second=known[key])

    # Première passe avec les seuils par défaut, uniquement pour mesurer le débit.
    draft = QualityProfile()
    long_enough = spread([s for s in segments if len(s.text) >= draft.min_chars], sample_size)
    measures: list[tuple[str, float]] = []
    for segment in long_enough:
        take, _ = render_with_fallback(backend.say, segment.text, draft)
        if take.clean:
            measures.append((segment.text, take.duration))

    rate = calibrate(measures)
    if rate is None:
        return draft

    known[key] = rate
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(known, indent=1), encoding="utf-8")
    return QualityProfile(chars_per_second=rate)
