# Interface web de voxlibris. Pas de CUDA, pas de moteur de synthèse : elle dépose des
# tâches et affiche leur avancement. C'est ce qui la fait tenir en quelques centaines de
# mégaoctets, là où l'atelier en pèse huit gigas.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# Le conteneur tourne sous l'UID de l'hôte (voir docker-compose.yml). Tout ce qu'il
# doit écrire vit sous /data, monté depuis l'hôte, ou sous /tmp.
RUN mkdir -p /data && chmod 777 /data

EXPOSE 8000
CMD ["uvicorn", "voxlibris.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
