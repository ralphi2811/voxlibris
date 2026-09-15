# Atelier de voxlibris : le seul conteneur qui porte les moteurs de synthèse et CUDA.
#
# L'image de base fournit torch compilé pour CUDA ; le réinstaller depuis PyPI donnerait
# une roue CPU. Voir worker.constraints.txt, qui verrouille exactement cela.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg : assemblage des MP3 et du M4B. espeak-ng : phonémisation du français pour
# Kokoro. fonts-dejavu : la police de la couche de texte posée sur un scan lu par
# RapidOCR (ingest/ocr.py). libsndfile : lecture des WAV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        espeak-ng \
        libsndfile1 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Les dépendances d'abord, sur un paquet vide : c'est la couche lourde — PyTorch, les
# moteurs — et elle ne dépend que de pyproject.toml. Le code arrive après, si bien
# qu'une modification de src/ ne la refait pas, ni ici ni dans l'intégration continue.
COPY pyproject.toml LICENSE ./
COPY docker/worker.constraints.txt ./constraints.txt
RUN touch README.md && mkdir -p src/voxlibris && touch src/voxlibris/__init__.py \
    && pip install -c constraints.txt ".[tts]"
COPY README.md ./
COPY src ./src
RUN pip install --no-deps .

# Les poids des modèles vont dans un volume nommé, pour survivre aux reconstructions.
# Le conteneur tourne sous l'UID de l'hôte, qui n'existe pas dans l'image : tout
# répertoire à écrire doit donc être ouvert à n'importe quel UID, et HOME doit pointer
# quelque part d'inscriptible — espeak-ng et matplotlib y déposent leur configuration.
ENV HF_HOME=/models/hf \
    TTS_HOME=/models/tts \
    PIPER_VOICE_DIR=/models/piper \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl \
    XDG_CONFIG_HOME=/tmp/config \
    XDG_CACHE_HOME=/tmp/cache \
    HOME=/tmp \
    VOXLIBRIS_DATA=/data

RUN mkdir -p /models/hf /models/tts /models/piper /data \
    && chmod -R 777 /models /data

# La licence CPML de XTTS n'est pas acceptée ici : c'est à l'utilisateur de poser
# COQUI_TOS_AGREED=1 dans son .env, en connaissance de cause. Voir NOTICE.md.
CMD ["python", "-m", "voxlibris.worker"]
