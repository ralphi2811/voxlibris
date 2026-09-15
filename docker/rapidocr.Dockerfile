# RapidOCR servi chez vous : la reconnaissance de caractères des PDF scannés sans couche
# de texte. Facultatif : voir le profil « rapidocr » du docker-compose.yml. Le serveur
# HTTP est le nôtre, docker/rapidocr-server.py, deux routes autour de l'API Python.
#
# Tout tourne sur processeur, sous ONNX Runtime : pas de carte graphique, pas de PyTorch,
# une image d'un peu moins d'un gigaoctet. Les modèles PP-OCR (détection, orientation,
# reconnaissance « latin ») sont téléchargés ici, à la construction, et embarqués : le
# conteneur n'a rien à chercher au premier démarrage, et fonctionne sans réseau.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# OpenCV, sous RapidOCR, réclame ces deux bibliothèques même en version « headless ».
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ARG RAPIDOCR_VERSION=3.9.2
ENV UV_LINK_MODE=copy
RUN uv venv /opt/rapidocr/.venv --python 3.12 \
    && uv pip install --python /opt/rapidocr/.venv \
        "rapidocr==${RAPIDOCR_VERSION}" "onnxruntime>=1.20" \
        "fastapi>=0.115" "uvicorn>=0.30" \
    && rm -rf /root/.cache/uv

ENV PATH=/opt/rapidocr/.venv/bin:$PATH \
    RAPIDOCR_MODELS=/opt/rapidocr/models \
    RAPIDOCR_LANG=latin \
    HOME=/tmp

# Les modèles, une fois pour toutes. Le chargement les télécharge s'ils manquent : on
# charge donc une fois ici, puis le dossier est ouvert à l'UID de l'hôte, inconnu de
# l'image, sous lequel le conteneur tournera.
COPY docker/rapidocr-server.py /opt/rapidocr/server.py
RUN python -c "import sys; sys.argv = ['x']; sys.path.insert(0, '/opt/rapidocr'); import server; server.load()" \
    && chmod -R a+rX /opt/rapidocr/models

EXPOSE 1921
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:1921/health', timeout=4)"

CMD ["python", "/opt/rapidocr/server.py"]
