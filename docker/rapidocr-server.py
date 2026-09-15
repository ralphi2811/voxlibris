"""Serveur de reconnaissance de caractères pour voxlibris, autour de RapidOCR.

Un PDF scanné sans couche de texte n'est qu'une suite d'images : avant tout, il faut y
lire les mots. RapidOCR (Apache 2.0) exécute les modèles PP-OCR de PaddleOCR sous ONNX
Runtime, sur processeur, en une seconde ou deux par page — et repère le texte posé sur
une illustration là où Tesseract ne rend que du bruit. Ce fichier le met derrière deux
routes, dans l'image du profil « rapidocr » du docker-compose.yml ; le client qui
l'appelle est `src/voxlibris/ingest/ocr.py`.

    GET  /health   les modèles sont-ils chargés, pour quelle écriture
    POST /ocr      corps : l'image d'une page (PNG ou JPEG) → les lignes lues, chacune
                   avec son quadrilatère en pixels et son score

Les modèles sont embarqués dans l'image au moment de sa construction : rien à
télécharger au premier démarrage, et le conteneur tourne sans accès au réseau. La
reconnaissance charge par défaut le modèle « latin », qui couvre le français avec ses
accents, l'espagnol, l'allemand, l'italien… Le modèle par défaut de RapidOCR, chinois et
anglais, ne lit pas un « é ».

Réglages, par variables d'environnement :
    RAPIDOCR_LANG     latin (ou ch, en, cyrillic, arabic… — voir RapidOCR)
    RAPIDOCR_MODELS   /opt/rapidocr/models
    RAPIDOCR_PORT     1921
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

log = logging.getLogger("rapidocr-server")

LANG = os.environ.get("RAPIDOCR_LANG", "latin")
MODELS = os.environ.get("RAPIDOCR_MODELS", "/opt/rapidocr/models")
PORT = int(os.environ.get("RAPIDOCR_PORT", "1921"))
# Une image de page de plus de vingt mégaoctets n'est pas une page.
MAX_BYTES = 20 * 2**20


class State:
    engine = None
    # ONNX Runtime tient en un seul fil ici : les pages font la queue.
    lock = threading.Lock()


state = State()


def load() -> None:
    """Charge détection, orientation et reconnaissance, dans l'écriture demandée."""
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

    started = time.monotonic()
    params = {
        "Global.model_root_dir": MODELS,
        "Global.log_level": "warning",
        "Rec.lang_type": LangRec(LANG),
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_type": ModelType.MOBILE,
    }
    # Le modèle chinois par défaut est celui de la version 6, sans équivalent « latin ».
    if LANG == "ch":
        params.pop("Rec.ocr_version")
        params.pop("Rec.model_type")
    state.engine = RapidOCR(params=params)
    log.info("modèles chargés en %.1f s (reconnaissance : %s)", time.monotonic() - started, LANG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load()
    yield


app = FastAPI(title="voxlibris · RapidOCR", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ready": state.engine is not None, "lang": LANG, "engine": "rapidocr"}


@app.post("/ocr")
async def ocr(request: Request) -> dict:
    """Les lignes d'une image de page, de haut en bas, telles que RapidOCR les lit."""
    if state.engine is None:
        raise HTTPException(503, "Modèles en cours de chargement.")
    body = await request.body()
    if not body:
        raise HTTPException(400, "Corps vide : envoyez l'image de la page.")
    if len(body) > MAX_BYTES:
        raise HTTPException(413, "Image trop lourde.")
    started = time.monotonic()
    with state.lock:
        try:
            result = state.engine(body)
        except Exception as error:  # image illisible, ou modèle pris en défaut
            raise HTTPException(400, f"Image illisible : {error}") from error
    lines = []
    if result.boxes is not None and result.txts is not None:
        for box, text, score in zip(result.boxes, result.txts, result.scores, strict=False):
            lines.append(
                {
                    "text": str(text),
                    "box": [[float(x), float(y)] for x, y in box],
                    "score": round(float(score), 3),
                }
            )
    height, width = result.img.shape[:2] if result.img is not None else (0, 0)
    elapsed = time.monotonic() - started
    log.info("%dx%d px → %d lignes en %.1f s", width, height, len(lines), elapsed)
    return {
        "lines": lines,
        "width": int(width),
        "height": int(height),
        "elapsed": round(elapsed, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
