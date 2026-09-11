# ZONOS2 (Zyphra) servi chez vous. Facultatif : voir le profil « zonos2 » du
# docker-compose.yml. Le serveur est celui du dépôt de Zyphra, à une révision fixée,
# installé tel quel avec son fichier de verrouillage : on ne choisit rien à sa place.
# Les poids (16 Go) sont téléchargés au premier démarrage dans le volume des modèles ;
# le dépôt Hugging Face est ouvert, aucun jeton n'est nécessaire.
#
# L'image de base porte nvcc : les noyaux d'attention se compilent à la demande, et un
# serveur qui échoue faute de compilateur ne le dit qu'au fond d'une trace.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# git : le dépôt. ffmpeg : le serveur décode par lui les extraits de voix déposés.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ARG ZONOS2_REPO=https://github.com/Zyphra/Zonos2.git
ARG ZONOS2_COMMIT=194c0a3
WORKDIR /opt/zonos2
RUN git clone --filter=blob:none "$ZONOS2_REPO" . && git checkout --quiet "$ZONOS2_COMMIT"

# Python géré par uv, hors de l'image de base qui n'en a pas. L'environnement est
# reproduit à l'identique depuis uv.lock : mêmes versions que chez Zyphra.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/zonos2/.venv \
    UV_LINK_MODE=copy
RUN uv python install 3.12 && uv sync --frozen --no-dev --python 3.12 \
    && rm -rf /root/.cache/uv

# Le conteneur tourne sous l'UID de l'hôte, inconnu de l'image : tout ce qui s'écrit
# doit l'être quelque part d'ouvert. Les poids vont dans le volume des modèles.
ENV HF_HOME=/models/hf \
    HOME=/tmp \
    PATH=/opt/zonos2/.venv/bin:$PATH
RUN mkdir -p /models/hf && chmod -R 777 /models

COPY docker/zonos2-entrypoint.sh /usr/local/bin/zonos2-entrypoint
RUN chmod 755 /usr/local/bin/zonos2-entrypoint

EXPOSE 1919
ENTRYPOINT ["zonos2-entrypoint"]
