"""Contrôle qualité de la synthèse : détecter les énoncés ratés, et les rejouer.

Un modèle autorégressif comme XTTS peut, sur un segment isolé, boucler, ajouter du babil
ou basculer dans une autre langue — le tout sans lever la moindre erreur. Le fichier
produit paraît normal ; seule l'écoute révèle le problème. Ce module mesure donc chaque
énoncé et rejoue ceux qui sortent des clous.

Trois défauts distincts sont recherchés, parce qu'aucun détecteur ne les couvre tous :

1. **Durée anormale**, rapportée au nombre de caractères. Attrape les emballements francs.
2. **Plafond absolu**, indépendant de la longueur. Le contrôle par ratio ne s'applique
   qu'au-dessus d'un seuil de caractères, et c'est précisément dans cet angle mort qu'une
   phrase de dix caractères a été rendue en plus de quatre secondes.
3. **Débordement de la dernière phrase**, mesuré séparément. Quand le babil s'enchaîne
   directement après le dernier mot, sans silence intermédiaire, la durée totale ne
   paraît que modérément longue alors que la fin est absurde.

Les seuils par défaut ont été calibrés sur une trentaine de tirages mesurés : les prises
saines tiennent entre 14 et 21 caractères par seconde, les tirages partis en vrille
tombent à 12 ou moins, et jusqu'à 3,6 pour une phrase courte qui part complètement
ailleurs. Le débit de référence, lui, dépend de la voix — voir `calibrate`.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 24000
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
SPLIT_PAUSE_MS = 250


@dataclass(frozen=True)
class QualityProfile:
    """Seuils du contrôle qualité.

    `chars_per_second` est la seule valeur réellement dépendante de la voix : une voix
    posée et une voix rapide n'ont pas le même débit, et l'utiliser telle quelle pour une
    autre voix fausserait tous les autres seuils, qui en dérivent.
    """

    chars_per_second: float = 18.0
    # Marge d'attaque et de chute. Faible, car `trim_edges` a déjà retiré les silences de
    # bord : la durée restante est presque entièrement de la parole.
    overhead_seconds: float = 0.15
    min_ratio: float = 0.75
    max_ratio: float = 1.45
    min_chars: int = 12
    # Aucune lecture plausible ne descend sous dix caractères par seconde.
    absolute_min_rate: float = 10.0
    absolute_overhead: float = 0.8
    max_attempts: int = 5
    trailing_gap_ms: int = 500
    trailing_max_ms: int = 400
    trailing_overrun_ratio: float = 1.6

    def expected(self, text: str) -> float:
        return len(text) / self.chars_per_second + self.overhead_seconds

    def hard_max(self, text: str) -> float:
        return len(text) / self.absolute_min_rate + self.absolute_overhead


def _envelope(audio: np.ndarray, threshold_db: float = -45.0) -> tuple[np.ndarray, int]:
    """Indices des fenêtres de 10 ms contenant de la parole, et la taille de fenêtre."""
    window = max(1, SAMPLE_RATE // 100)
    usable = len(audio) - len(audio) % window
    if usable < window:
        return np.array([], dtype=int), window
    frames = audio[:usable].reshape(-1, window)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-12
    return np.flatnonzero(20 * np.log10(rms) > threshold_db), window


def trim_edges(audio: np.ndarray, keep_ms: int = 40) -> np.ndarray:
    """Retire le silence et le souffle en tête et en queue d'un énoncé.

    Sans ce rognage, les pauses valent « ce qui est calculé, plus ce que le modèle a bien
    voulu laisser », et le rythme du livre devient irrégulier.
    """
    if audio.size == 0:
        return audio
    loud, window = _envelope(audio)
    if loud.size == 0:
        return audio
    keep = max(1, keep_ms * SAMPLE_RATE // 1000 // window)
    start = max(0, loud[0] - keep) * window
    end = min(len(audio) // window, loud[-1] + 1 + keep) * window
    return audio[start:end]


def last_speech_run(audio: np.ndarray, gap_ms: int = 200) -> float:
    """Durée de la dernière plage de parole, délimitée par le dernier vrai silence."""
    loud, window = _envelope(audio)
    if loud.size < 2:
        return len(audio) / SAMPLE_RATE
    big = np.flatnonzero(np.diff(loud) >= gap_ms // 10)
    if big.size == 0:
        return len(audio) / SAMPLE_RATE
    return (loud[-1] - loud[big[-1] + 1] + 1) * window / SAMPLE_RATE


def trailing_artifact(audio: np.ndarray, profile: QualityProfile) -> bool:
    """Vrai si un résidu sonore bref traîne après un silence en fin d'énoncé."""
    loud, window = _envelope(audio)
    if loud.size < 2:
        return False
    big = np.flatnonzero(np.diff(loud) >= profile.trailing_gap_ms // 10)
    if big.size == 0:
        return False
    # Seul le dernier trou compte : les précédents sont les pauses normales entre les
    # phrases d'un même segment. Et l'on n'incrimine que si ce qui suit est bref — une
    # vraie fin de phrase dure plus longtemps qu'un résidu.
    tail = loud[-1] - loud[big[-1] + 1] + 1
    return tail * (window * 1000 // SAMPLE_RATE) <= profile.trailing_max_ms


def trailing_overrun(audio: np.ndarray, text: str, profile: QualityProfile) -> bool:
    """Vrai si la dernière phrase déborde de sa durée plausible.

    Ne s'applique qu'aux textes de plusieurs phrases : sans silence interne, il n'y a
    aucun repère pour délimiter la dernière plage.
    """
    sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) < 2:
        return False
    return last_speech_run(audio) > profile.expected(sentences[-1]) * profile.trailing_overrun_ratio


def inspect(audio: np.ndarray, text: str, profile: QualityProfile) -> str:
    """Renvoie la cause du rejet, ou une chaîne vide si l'énoncé paraît sain."""
    duration = len(audio) / SAMPLE_RATE
    if duration > profile.hard_max(text):
        return "durée absurde"
    expected = profile.expected(text)
    if len(text) >= profile.min_chars and not (
        expected * profile.min_ratio <= duration <= expected * profile.max_ratio
    ):
        return "durée"
    if trailing_artifact(audio, profile):
        return "résidu en fin d'énoncé"
    if trailing_overrun(audio, text, profile):
        return "dernière phrase trop longue"
    return ""


@dataclass
class Take:
    """Résultat d'un rendu : l'audio retenu, son état et le nombre d'essais consentis."""

    audio: np.ndarray
    clean: bool
    attempts: int
    cause: str = ""

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


def _supersedes(candidate: Take, incumbent: Take | None, expected_samples: float) -> bool:
    """Une prise saine l'emporte toujours ; sinon la plus proche de la durée attendue."""
    if incumbent is None:
        return True
    if candidate.clean != incumbent.clean:
        return candidate.clean
    return abs(len(candidate.audio) - expected_samples) < abs(
        len(incumbent.audio) - expected_samples
    )


def render(say: Callable[[str], np.ndarray], text: str, profile: QualityProfile) -> Take:
    """Synthétise un texte, en rejouant tant que le résultat paraît fautif.

    XTTS étant stochastique, un énoncé parti en vrille se rattrape presque toujours au
    tirage suivant. À défaut d'une prise saine, on conserve celle dont la durée s'écarte
    le moins de l'attendu, plutôt que la dernière venue.
    """
    expected_samples = profile.expected(text) * SAMPLE_RATE
    best: Take | None = None
    attempt = 0

    for attempt in range(1, profile.max_attempts + 1):
        audio = trim_edges(say(text))
        cause = inspect(audio, text, profile)
        take = Take(audio=audio, clean=not cause, attempts=attempt, cause=cause)
        if _supersedes(take, best, expected_samples):
            best = take
        if take.clean:
            break

    assert best is not None
    # `attempts` compte les essais consentis, pas le rang de la prise retenue.
    best.attempts = attempt
    return best


def render_with_fallback(
    say: Callable[[str], np.ndarray], text: str, profile: QualityProfile
) -> tuple[Take, bool]:
    """Comme `render`, mais en dernier recours rejoue le texte phrase par phrase.

    Certains enchaînements de phrases font échouer le modèle de façon reproductible :
    insister ne sert à rien, alors que les mêmes phrases prises isolément passent sans
    peine. Le second élément du couple indique si ce repli a été employé.
    """
    take = render(say, text, profile)
    sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if take.clean or len(sentences) < 2:
        return take, False

    parts: list[np.ndarray] = []
    total, ok = take.attempts, True
    pause = np.zeros(int(SAMPLE_RATE * SPLIT_PAUSE_MS / 1000), np.float32)
    for sentence in sentences:
        sub = render(say, sentence, profile)
        total += sub.attempts
        ok &= sub.clean
        parts += [sub.audio, pause]

    candidate = np.concatenate(parts[:-1])
    # Le plafond absolu s'applique aussi au recollement : chaque phrase peut être jugée
    # correcte isolément alors que leur somme reste aberrante.
    if ok and len(candidate) / SAMPLE_RATE <= profile.hard_max(text):
        return Take(candidate, True, total), True
    return take, False


def calibrate(samples: list[tuple[str, float]], floor: int = 8) -> float | None:
    """Déduit le débit d'une voix à partir d'énoncés déjà produits.

    `samples` associe un texte à la durée mesurée de son rendu. La médiane est retenue :
    elle ignore les quelques tirages ratés qui traînent toujours dans un lot, là où une
    moyenne s'en trouverait tirée vers le bas.

    Renvoie None si l'échantillon est trop maigre pour être significatif.
    """
    rates = [len(text) / duration for text, duration in samples if duration > 0 and text]
    if len(rates) < floor:
        return None
    return round(statistics.median(rates), 1)
