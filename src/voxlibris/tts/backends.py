"""Moteurs de synthèse vocale, derrière une interface commune.

Chaque moteur se réduit à `say(texte) -> forme d'onde` et à la liste de ses voix. Les
imports sont volontairement locaux à chaque constructeur : installer Piper ne doit pas
obliger à installer la pile CUDA de XTTS, et le module doit rester importable même quand
aucun moteur n'est présent — l'interface web en a besoin pour afficher les choix.

Licences des modèles, très différentes les unes des autres, voir NOTICE.md :
  XTTS-v2  CPML, usage non commercial
  Kokoro   Apache 2.0
  Piper    MIT
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000

# Paramètres de génération XTTS, choisis en mesurant cinq tirages par configuration sur
# un texte qui déclenchait un défaut une fois sur deux :
#   temperature 0.75 (défaut du modèle) : 2 tirages fautifs sur 5 ;
#   temperature 0.60                    : 0 sur 5, durées resserrées ;
#   temperature 0.45                    : dégradation nette, jusqu'à 14,75 s pour 5,3 s.
# split_sentences=False désactive la découpe interne de TTS.api, qui segmentait le texte
# à notre insu et recollait les morceaux avec ses propres silences.
XTTS_GEN_PARAMS = {"temperature": 0.60, "split_sentences": False}

KOKORO_VOICE = "ff_siwis"
PIPER_VOICES = ["fr_FR-tom-medium", "fr_FR-upmc-medium", "fr_FR-siwis-medium"]


def resample(audio: np.ndarray, source_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Rééchantillonnage linéaire, suffisant pour homogénéiser les sorties des moteurs."""
    if source_rate == target_rate:
        return audio
    target_len = int(len(audio) / source_rate * target_rate)
    return np.interp(
        np.linspace(0, len(audio) - 1, target_len), np.arange(len(audio)), audio
    ).astype(np.float32)


# Débit de parole. 1.0 est le débit natif du modèle ; en dessous, la lecture ralentit.
# Un livre audio s'écoute plus lentement qu'une phrase de démonstration, et les moteurs
# sont réglés pour la seconde. Chaque moteur exprime cela à sa façon — facteur direct
# chez XTTS et Kokoro, échelle de durée inversée chez Piper — d'où cette conversion.
#
# Mesuré sur XTTS, cinq tirages par réglage sur une même phrase (durée médiane) :
#   1.00× → 3.76 s   tirages resserrés sur 0,27 s
#   0.80× → 4.68 s   soit 1.24× plus long, conforme à 1/0.8
#   0.65× → 5.65 s   conforme aussi, mais la dispersion triple (5,38 à 6,91 s)
# C'est cette dispersion qui borne la plage : plus bas, le modèle devient instable et le
# contrôle qualité rejouerait sans fin. Un seul tirage ne montre rien de tout cela — la
# variance naturelle du modèle dépasse l'effet cherché.
DEFAULT_SPEED = 1.0
SPEED_RANGE = (0.7, 1.3)


def clamp_speed(speed: float) -> float:
    low, high = SPEED_RANGE
    return max(low, min(high, float(speed)))


def resolve_voice(wanted: str, catalogue: Sequence[str]) -> str | None:
    """Retrouve une voix du catalogue à partir de ce qui a été saisi.

    Les identifiants sont longs et préfixés — « fr_FR-tom-medium » — et rien n'invite à
    les recopier au caractère près. « tom », « FR-tom-medium » ou « Fr_FR-Tom-Medium »
    désignent tous la même voix sans la moindre ambiguïté ; les refuser serait de la
    pédanterie. On n'accepte en revanche que les correspondances uniques : si la saisie
    convient à deux voix, c'est à l'utilisateur de trancher, pas à nous de deviner.
    """
    if wanted in catalogue:
        return wanted
    folded = wanted.strip().lower()
    if not folded:
        return None
    for test in (lambda v: v.lower() == folded, lambda v: folded in v.lower()):
        matches = [voice for voice in catalogue if test(voice)]
        if len(matches) == 1:
            return matches[0]
    return None


class UnknownVoice(RuntimeError):
    """Voix inconnue du moteur retenu.

    Le cas se produit sitôt qu'on change de moteur sans changer de voix, et ce qu'en
    disent les moteurs est incompréhensible : Kokoro va chercher « Damien Black.pt » sur
    un dépôt de modèles et rapporte une erreur 404. Autant nommer la vraie cause, et
    donner les voix qui, elles, existent.
    """

    def __init__(self, backend: str, voice: str, available: list[str]) -> None:
        shown = ", ".join(available[:12]) + (" …" if len(available) > 12 else "")
        super().__init__(
            f"Le moteur {backend!r} ne connaît pas la voix {voice!r}. "
            f"Voix disponibles : {shown}"
        )


class Backend:
    """Interface commune aux moteurs."""

    name = "base"
    sample_rate = SAMPLE_RATE
    speed = DEFAULT_SPEED
    # Voix connues d'avance, quand elles le sont. XTTS laisse ce catalogue vide : ses
    # locuteurs ne se lisent qu'une fois le modèle chargé, et la vérification se fait
    # alors dans son constructeur.
    catalogue: tuple[str, ...] = ()

    def say(self, text: str) -> np.ndarray:
        """Synthétise un texte et renvoie une forme d'onde mono à SAMPLE_RATE."""
        raise NotImplementedError

    @staticmethod
    def voices() -> list[str]:
        raise NotImplementedError


class XttsBackend(Backend):
    """XTTS-v2 — meilleure prosodie en français, GPU, licence CPML non commerciale."""

    name = "xtts"
    default_voice = "Viktor Menelaos"

    def __init__(
        self,
        voice: str = default_voice,
        device: str = "cuda",
        speed: float = DEFAULT_SPEED,
    ) -> None:
        from TTS.api import TTS

        if not os.environ.get("COQUI_TOS_AGREED"):
            raise RuntimeError(
                "XTTS-v2 est sous licence CPML (usage non commercial). Positionnez "
                "COQUI_TOS_AGREED=1 pour confirmer que vous l'acceptez, ou choisissez "
                "un autre moteur (kokoro, piper). Voir NOTICE.md."
            )
        self.voice = voice
        self.speed = clamp_speed(speed)
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.sample_rate = self._tts.synthesizer.output_sample_rate
        # Le catalogue de XTTS n'est lisible qu'ici, le modèle une fois en mémoire.
        resolved = resolve_voice(voice, self.voices())
        if resolved is None:
            raise UnknownVoice(self.name, voice, self.voices())
        self.voice = resolved

    def voices(self) -> list[str]:  # type: ignore[override]
        return sorted(self._tts.synthesizer.tts_model.speaker_manager.speakers.keys())

    def say(self, text: str) -> np.ndarray:
        wav = self._tts.tts(
            text=text, speaker=self.voice, language="fr", speed=self.speed, **XTTS_GEN_PARAMS
        )
        return resample(np.asarray(wav, dtype=np.float32), self.sample_rate)


class KokoroBackend(Backend):
    """Kokoro-82M — Apache 2.0, rapide et très stable, une seule voix française."""

    name = "kokoro"
    default_voice = KOKORO_VOICE
    catalogue = (KOKORO_VOICE,)

    def __init__(
        self,
        voice: str = KOKORO_VOICE,
        device: str | None = None,
        speed: float = DEFAULT_SPEED,
    ) -> None:
        from kokoro import KPipeline

        self.voice = voice
        self.speed = clamp_speed(speed)
        self._pipeline = KPipeline(lang_code="f", device=device)
        self.sample_rate = 24000

    @staticmethod
    def voices() -> list[str]:
        return [KOKORO_VOICE]

    def say(self, text: str) -> np.ndarray:
        # split_pattern est neutralisé : la découpe est faite en amont et doit le rester,
        # sinon les pauses calculées ne correspondent plus aux segments.
        parts = [
            result.output.audio.detach().cpu().numpy()
            for result in self._pipeline(
                text, voice=self.voice, speed=self.speed, split_pattern=None
            )
            if result.output is not None and result.output.audio is not None
        ]
        if not parts:
            return np.zeros(0, np.float32)
        return resample(np.concatenate(parts).astype(np.float32), self.sample_rate)


class PiperBackend(Backend):
    """Piper — MIT, processeur, quasi instantané. Idéal pour un brouillon d'écoute."""

    name = "piper"
    default_voice = PIPER_VOICES[0]
    catalogue = tuple(PIPER_VOICES)

    def __init__(
        self,
        voice: str = PIPER_VOICES[0],
        device: str | None = None,
        speed: float = DEFAULT_SPEED,
    ) -> None:
        from piper import PiperVoice
        from piper.download_voices import download_voice

        directory = Path(os.environ.get("PIPER_VOICE_DIR", "~/.cache/piper")).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        # PiperVoice.load attend un chemin de fichier, pas un nom de voix : le
        # téléchargement doit être déclenché explicitement au premier usage.
        model = directory / f"{voice}.onnx"
        if not model.exists():
            download_voice(voice, directory)

        self.voice = voice
        self.speed = clamp_speed(speed)
        self._voice = PiperVoice.load(model)
        self.sample_rate = self._voice.config.sample_rate

    @staticmethod
    def voices() -> list[str]:
        return list(PIPER_VOICES)

    def say(self, text: str) -> np.ndarray:
        # Piper raisonne en échelle de durée : allonger les phonèmes ralentit la parole.
        from piper import SynthesisConfig

        config = SynthesisConfig(length_scale=1.0 / self.speed)
        chunks = [chunk.audio_float_array for chunk in self._voice.synthesize(text, config)]
        if not chunks:
            return np.zeros(0, np.float32)
        return resample(np.concatenate(chunks).astype(np.float32), self.sample_rate)


BACKENDS: dict[str, type[Backend]] = {
    "xtts": XttsBackend,
    "kokoro": KokoroBackend,
    "piper": PiperBackend,
}


def load(
    name: str,
    voice: str | None = None,
    device: str = "cuda",
    speed: float = DEFAULT_SPEED,
) -> Backend:
    """Instancie un moteur par son nom, avec sa voix par défaut si aucune n'est donnée.

    Les moteurs vivent dans l'extra « tts », absent d'une installation ordinaire — et
    qu'un simple `uv sync` désinstalle au passage. L'erreur brute d'import ne nomme que
    le module manquant, jamais le remède : on la traduit ici, une fois pour toutes.
    """
    if name not in BACKENDS:
        raise ValueError(f"Moteur inconnu : {name!r}. Disponibles : {', '.join(BACKENDS)}")
    cls = BACKENDS[name]
    wanted = voice or cls.default_voice  # type: ignore[attr-defined]
    # Vérifié avant d'instancier quand le catalogue est connu : charger huit gigaoctets
    # de modèle pour découvrir ensuite que la voix n'existe pas serait une perte de temps.
    if cls.catalogue:
        resolved = resolve_voice(wanted, cls.catalogue)
        if resolved is None:
            raise UnknownVoice(name, wanted, list(cls.catalogue))
        wanted = resolved
    try:
        return cls(wanted, device, speed)  # type: ignore[call-arg]
    except ImportError as error:
        raise RuntimeError(
            f"Le moteur {name!r} n'est pas installé ({error.name} manquant). "
            "Installez les moteurs de synthèse avec « uv sync --extra tts », ou "
            "utilisez l'image Docker de l'ouvrier, qui les embarque."
        ) from error
