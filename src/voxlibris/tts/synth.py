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
from .quality import SAMPLE_RATE, QualityProfile, calibrate, render_with_fallback


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

    def write(self, target: Path) -> None:
        import soundfile as sf

        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, self.audio, SAMPLE_RATE)
        target.with_suffix(".timing.json").write_text(
            json.dumps(self.timing, ensure_ascii=False, indent=1), encoding="utf-8"
        )


def synthesize_chapter(
    backend: Backend,
    segments: Iterable[Segment],
    profile: QualityProfile | None = None,
    on_segment: Callable[[Segment, float, int], None] = lambda *_: None,
) -> ChapterResult:
    """Synthétise et concatène un chapitre, en signalant les segments douteux."""
    profile = profile or QualityProfile()
    pieces: list[np.ndarray] = []
    timing: list[dict] = []
    warnings: list[str] = []
    cursor = 0

    for segment in segments:
        take, split = render_with_fallback(backend.say, segment.text, profile)

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
                "split": split,
                "text": segment.text,
            }
        )
        cursor += len(take.audio) + len(pause)
        pieces += [take.audio, pause]
        on_segment(segment, take.duration, take.attempts)

    audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    return ChapterResult(audio=audio, timing=timing, warnings=warnings)


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
    key = f"{backend.name}/{getattr(backend, 'voice', 'default')}"
    known = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
    if key in known:
        return QualityProfile(chars_per_second=known[key])

    # Première passe avec les seuils par défaut, uniquement pour mesurer le débit.
    draft = QualityProfile()
    long_enough = [s for s in segments if len(s.text) >= draft.min_chars][:sample_size]
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
