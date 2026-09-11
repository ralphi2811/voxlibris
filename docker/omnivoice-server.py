"""Serveur HTTP d'OmniVoice pour voxlibris.

Le paquet `omnivoice` (k2-fsa, Apache 2.0) n'expose qu'une API Python : ce fichier la
met derrière trois routes, dans l'image du profil « omnivoice » du docker-compose.yml.
C'est le seul code de voxlibris qui tourne dans ce conteneur ; le client qui l'appelle
est `src/voxlibris/tts/omnivoice.py`.

    GET  /health   le modèle est-il chargé, sur quoi, à quelle fréquence
    GET  /voices   les voix : un fichier audio du dossier des voix = une voix
    POST /speak    {text, voice, language?, speed?} → PCM flottant 32 bits mono,
                   fréquence dans l'en-tête X-Audio-Sample-Rate

Une voix se prépare une fois : l'extrait est coupé à une douzaine de secondes au
silence le plus net (au-delà, le modèle clone moins bien et va moins vite), transcrit
par Whisper, puis encodé en jetons acoustiques. Cette « invite de clonage » reste en
mémoire tant que le fichier ne change pas ; le dossier est relu à chaque liste, une voix
déposée existe aussitôt. Le même dossier sert à ZONOS2.

Réglages, par variables d'environnement :
    OMNIVOICE_MODEL            k2-fsa/OmniVoice
    OMNIVOICE_ASR              openai/whisper-large-v3-turbo (transcription des extraits)
    OMNIVOICE_ASR_DEVICE       cuda ou cpu ; par défaut, cpu sous 12 Go de carte
    OMNIVOICE_VOICES_DIR       /data/voices
    OMNIVOICE_MAX_REFERENCE_S  12
    OMNIVOICE_STEPS            32 (16 va deux fois plus vite, un peu moins bien)
    OMNIVOICE_PORT             1920
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

log = logging.getLogger("omnivoice-server")

MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
ASR_ID = os.environ.get("OMNIVOICE_ASR", "openai/whisper-large-v3-turbo")
VOICES_DIR = Path(os.environ.get("OMNIVOICE_VOICES_DIR", "/data/voices"))
MAX_REFERENCE_S = float(os.environ.get("OMNIVOICE_MAX_REFERENCE_S", "12"))
NUM_STEP = int(os.environ.get("OMNIVOICE_STEPS", "32"))
PORT = int(os.environ.get("OMNIVOICE_PORT", "1920"))

# Ce que le dossier des voix peut contenir ; même liste que l'atelier et que ZONOS2.
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".webm")


class State:
    model = None
    device = ""
    sample_rate = 24000
    # Une seule carte, une génération à la fois : les requêtes font la queue.
    lock = threading.Lock()
    # nom de fichier → (mtime, taille, invite, transcription, secondes retenues)
    prompts: dict[str, tuple[float, int, object, str, float]] = {}


state = State()


def voice_id(filename: str) -> str:
    """Empreinte du nom de fichier : stable, sans espace ni accent, comme chez ZONOS2."""
    return "voice_" + hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]


def label_of(filename: str) -> str:
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").strip() or filename


def scan() -> list[dict[str, str]]:
    """Les voix, dans l'ordre du dossier — relu à chaque appel."""
    if not VOICES_DIR.is_dir():
        return []
    return [
        {"id": voice_id(path.name), "label": label_of(path.name), "file": path.name}
        for path in sorted(VOICES_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]


def find_voice(wanted: str) -> Path:
    """Par identifiant, par nom de fichier ou par intitulé, sans casse."""
    wanted = wanted.strip()
    for voice in scan():
        if wanted in (voice["id"], voice["file"]) or wanted.lower() == voice["label"].lower():
            return VOICES_DIR / voice["file"]
    raise HTTPException(404, f"Voix inconnue : {wanted!r}. Voix : {[v['label'] for v in scan()]}")


def release_asr() -> None:
    """Décharge Whisper de la carte : il ne sert qu'à préparer une voix, c'est rare.

    Il sera rechargé à la prochaine voix déposée, en quelques secondes. Le laisser
    reviendrait à occuper deux gigaoctets pour rien pendant toute la lecture.
    """
    import torch

    if getattr(state.model, "_asr_pipe", None) is None:
        return
    state.model._asr_pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def free_cache() -> None:
    """Rend à la carte ce que PyTorch garde en réserve entre deux phrases.

    L'allocateur conserve sinon le pic de la génération précédente ; rendu à chaque fois,
    le serveur tient dans l'espace annoncé, à côté d'un autre moteur.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prompt_for(path: Path, keep_asr: bool = False):
    """L'invite de clonage du fichier, préparée à la première demande ou s'il a changé."""
    stat = path.stat()
    cached = state.prompts.get(path.name)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]
    from omnivoice.utils.audio import load_audio, trim_long_audio

    started = time.monotonic()
    rate = state.sample_rate
    wav = load_audio(str(path), rate)
    # Coupe au début d'une phrase, jamais au milieu d'un mot : la transcription qui suit
    # doit coller à ce que le modèle entend.
    wav = trim_long_audio(
        wav, rate, max_duration=MAX_REFERENCE_S, min_duration=3.0, trim_threshold=MAX_REFERENCE_S
    )
    prompt = state.model.create_voice_clone_prompt((wav, rate))
    seconds = wav.shape[-1] / rate
    log.info(
        "voix « %s » : %.1f s retenues, transcrite en %.0f s : %s",
        label_of(path.name),
        seconds,
        time.monotonic() - started,
        prompt.ref_text[:100],
    )
    state.prompts[path.name] = (stat.st_mtime, stat.st_size, prompt, prompt.ref_text, seconds)
    if not keep_asr:
        release_asr()
    return prompt


def load_model() -> None:
    import torch
    from omnivoice import OmniVoice
    from omnivoice.utils.common import get_best_device

    device = os.environ.get("OMNIVOICE_DEVICE") or get_best_device()
    dtype = torch.float16 if device.startswith(("cuda", "xpu")) else torch.float32
    # La lecture tient en trois gigaoctets ; Whisper, lui, en prend dix le temps de
    # transcrire un extrait. Sur une petite carte, il passe sur le processeur : plus lent,
    # mais un dépôt de voix est rare.
    asr_device = os.environ.get("OMNIVOICE_ASR_DEVICE") or device
    if device.startswith("cuda") and not os.environ.get("OMNIVOICE_ASR_DEVICE"):
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total_gb < 12:
            asr_device = "cpu"
    log.info("chargement de %s sur %s (Whisper sur %s)…", MODEL_ID, device, asr_device)
    started = time.monotonic()
    state.model = OmniVoice.from_pretrained(
        MODEL_ID, device_map=device, dtype=dtype, asr_model_name=ASR_ID, asr_device=asr_device
    )
    state.device = device
    state.sample_rate = int(state.model.sampling_rate)
    log.info(
        "modèle chargé en %.0f s, sortie à %d Hz", time.monotonic() - started, state.sample_rate
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    load_model()
    # Les voix présentes sont préparées d'emblée : la première phrase ne paie pas Whisper.
    for voice in scan():
        try:
            with state.lock:
                prompt_for(VOICES_DIR / voice["file"], keep_asr=True)
        except Exception as error:  # un extrait illisible ne doit pas empêcher les autres
            log.warning("voix « %s » ignorée : %s", voice["label"], error)
    with state.lock:
        release_asr()
    log.info("prêt ; préparation des voix : %s", vram())
    # Le pic repart de zéro : celui qu'on annonce ensuite est celui de la lecture, pas
    # celui de Whisper, qui n'a lieu qu'au dépôt d'une voix.
    reset_peak()
    yield


app = FastAPI(title="voxlibris · OmniVoice", lifespan=lifespan)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    # Code ISO à deux lettres ; absent, le modèle devine d'après le texte.
    language: str | None = None
    speed: float = Field(default=1.0, gt=0.25, lt=4.0)
    num_step: int | None = Field(default=None, ge=4, le=64)


def vram() -> dict[str, float]:
    """Ce que le serveur tient sur la carte, en Go : alloué, et réservé par PyTorch."""
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "vram_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
        "vram_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
        "vram_peak_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }


@app.get("/health")
def health() -> dict:
    return {
        "ready": state.model is not None,
        "model": MODEL_ID,
        "device": state.device,
        "sample_rate": state.sample_rate,
        "voices": len(scan()),
        "max_reference_s": MAX_REFERENCE_S,
        **vram(),
    }


@app.get("/voices")
def voices() -> dict:
    found = scan()
    for voice in found:
        cached = state.prompts.get(voice["file"])
        voice["prepared"] = bool(cached)
        if cached:
            voice["reference_s"] = round(cached[4], 1)
    return {"voices": found}


@app.post("/speak")
def speak(request: SpeakRequest) -> Response:
    if state.model is None:
        raise HTTPException(503, "Modèle en cours de chargement.")
    path = find_voice(request.voice)
    language = (request.language or "").strip().lower()[:2] or None
    if language and language not in state.model.supported_language_ids():
        log.warning("langue %r inconnue du modèle : détection automatique", language)
        language = None
    with state.lock:
        prompt = prompt_for(path)
        started = time.monotonic()
        try:
            audios = state.model.generate(
                text=request.text,
                language=language,
                voice_clone_prompt=prompt,
                speed=request.speed if request.speed != 1.0 else None,
                num_step=request.num_step or NUM_STEP,
            )
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        free_cache()
    audio = np.asarray(audios[0], dtype=np.float32).reshape(-1)
    seconds = len(audio) / state.sample_rate
    log.info(
        "%d caractères → %.1f s d'audio en %.1f s",
        len(request.text),
        seconds,
        time.monotonic() - started,
    )
    return Response(
        content=audio.astype("<f4").tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Sample-Rate": str(state.sample_rate),
            "X-Audio-Seconds": f"{seconds:.2f}",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
