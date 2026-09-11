# OmniVoice (k2-fsa) servi chez vous. Facultatif : voir le profil « omnivoice » du
# docker-compose.yml. Le paquet ne livre pas de serveur HTTP : celui-ci est le nôtre,
# docker/omnivoice-server.py, trois routes de FastAPI autour de l'API Python.
#
# Pas de CUDA dans l'image de base : les roues PyTorch de PyPI embarquent leurs
# bibliothèques, seul le pilote de la machine est requis, par le nvidia-container-toolkit.
# Les poids (1,6 Go, plus Whisper pour transcrire les extraits) se téléchargent au
# premier démarrage dans le volume des modèles ; dépôt ouvert, aucun jeton.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ffmpeg : les extraits de voix en mp3, m4a ou ogg se décodent par lui.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

# Versions fixées : le paquet, et PyTorch à la même version que l'ouvrier de l'atelier.
ARG OMNIVOICE_VERSION=0.2.1
ARG TORCH_VERSION=2.9.1
ENV UV_LINK_MODE=copy
RUN uv venv /opt/omnivoice/.venv --python 3.12 \
    && uv pip install --python /opt/omnivoice/.venv \
        "omnivoice==${OMNIVOICE_VERSION}" \
        "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}" \
        "fastapi>=0.115" "uvicorn>=0.30" \
    && rm -rf /root/.cache/uv

# Le conteneur tourne sous l'UID de l'hôte, inconnu de l'image : tout ce qui s'écrit
# doit l'être quelque part d'ouvert. Les poids vont dans le volume des modèles.
ENV HF_HOME=/models/hf \
    HOME=/tmp \
    PATH=/opt/omnivoice/.venv/bin:$PATH
RUN mkdir -p /models/hf && chmod -R 777 /models

COPY docker/omnivoice-server.py /opt/omnivoice/server.py

EXPOSE 1920
CMD ["python", "/opt/omnivoice/server.py"]
