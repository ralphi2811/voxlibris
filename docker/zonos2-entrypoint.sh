#!/bin/sh
# Lance le serveur ZONOS2 sur le dossier des voix de l'atelier.
#
# Les voix sont les fichiers audio de ce dossier : en déposer un, c'est créer une voix.
# L'atelier y écrit depuis sa page Voix. Vide au premier démarrage, il reçoit les trois
# voix anglaises livrées avec le dépôt, pour que le moteur réponde tout de suite.
set -e

VOICES="${ZONOS2_VOICES_DIR:-/data/voices}"
mkdir -p "$VOICES"
if [ -z "$(ls -A "$VOICES" 2>/dev/null)" ]; then
    cp /opt/zonos2/default_voices/*.mp3 "$VOICES"/ 2>/dev/null || true
fi

# Le dossier de travail reste celui du dépôt : le serveur y trouve les directions
# d'émotion, qu'il charge de lui-même.
cd /opt/zonos2
exec python -m zonos2 \
    --model-path "${ZONOS2_MODEL:-Zyphra/ZONOS2}" \
    --host 0.0.0.0 --port 1919 \
    --tts-default-voices-dir "$VOICES" \
    "$@"
