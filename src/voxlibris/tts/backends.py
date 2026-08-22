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


class Backend:
    """Interface commune aux moteurs."""

    name = "base"
    sample_rate = SAMPLE_RATE

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

    def __init__(self, voice: str = default_voice, device: str = "cuda") -> None:
        from TTS.api import TTS

        if not os.environ.get("COQUI_TOS_AGREED"):
            raise RuntimeError(
                "XTTS-v2 est sous licence CPML (usage non commercial). Positionnez "
                "COQUI_TOS_AGREED=1 pour confirmer que vous l'acceptez, ou choisissez "
                "un autre moteur (kokoro, piper). Voir NOTICE.md."
            )
        self.voice = voice
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.sample_rate = self._tts.synthesizer.output_sample_rate

    def voices(self) -> list[str]:  # type: ignore[override]
        return sorted(self._tts.synthesizer.tts_model.speaker_manager.speakers.keys())

    def say(self, text: str) -> np.ndarray:
        wav = self._tts.tts(text=text, speaker=self.voice, language="fr", **XTTS_GEN_PARAMS)
        return resample(np.asarray(wav, dtype=np.float32), self.sample_rate)


class KokoroBackend(Backend):
    """Kokoro-82M — Apache 2.0, rapide et très stable, une seule voix française."""

    name = "kokoro"
    default_voice = KOKORO_VOICE

    def __init__(self, voice: str = KOKORO_VOICE, device: str | None = None) -> None:
        from kokoro import KPipeline

        self.voice = voice
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
            for result in self._pipeline(text, voice=self.voice, split_pattern=None)
            if result.output is not None and result.output.audio is not None
        ]
        if not parts:
            return np.zeros(0, np.float32)
        return resample(np.concatenate(parts).astype(np.float32), self.sample_rate)


class PiperBackend(Backend):
    """Piper — MIT, processeur, quasi instantané. Idéal pour un brouillon d'écoute."""

    name = "piper"
    default_voice = PIPER_VOICES[0]

    def __init__(self, voice: str = PIPER_VOICES[0], device: str | None = None) -> None:
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
        self._voice = PiperVoice.load(model)
        self.sample_rate = self._voice.config.sample_rate

    @staticmethod
    def voices() -> list[str]:
        return list(PIPER_VOICES)

    def say(self, text: str) -> np.ndarray:
        chunks = [chunk.audio_float_array for chunk in self._voice.synthesize(text)]
        if not chunks:
            return np.zeros(0, np.float32)
        return resample(np.concatenate(chunks).astype(np.float32), self.sample_rate)


BACKENDS: dict[str, type[Backend]] = {
    "xtts": XttsBackend,
    "kokoro": KokoroBackend,
    "piper": PiperBackend,
}


def load(name: str, voice: str | None = None, device: str = "cuda") -> Backend:
    """Instancie un moteur par son nom, avec sa voix par défaut si aucune n'est donnée."""
    if name not in BACKENDS:
        raise ValueError(f"Moteur inconnu : {name!r}. Disponibles : {', '.join(BACKENDS)}")
    cls = BACKENDS[name]
    return cls(voice or cls.default_voice, device)  # type: ignore[call-arg,attr-defined]
