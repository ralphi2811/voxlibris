# Voxtral TTS servi par vLLM, sur la machine. Facultatif : voir le profil « voxtral »
# du docker-compose.yml. Les poids (8 Go) sont téléchargés au premier démarrage dans
# le volume des modèles ; le dépôt est sous conditions, il faut un jeton Hugging Face
# dont le compte a accepté la licence CC BY-NC 4.0.
FROM vllm/vllm-openai:v0.28.0

# vllm-omni ajoute les modèles audio à vLLM ; ninja sert à compiler des noyaux au
# démarrage — sans lui, l'échec se lit à cinquante lignes de sa cause.
RUN pip install --no-cache-dir "vllm-omni==0.28.0" ninja

ENV HF_HOME=/models/hf \
    HOME=/tmp
RUN mkdir -p /models/hf && chmod -R 777 /models

ENTRYPOINT ["vllm", "serve", "mistralai/Voxtral-4B-TTS-2603", "--omni", "--host", "0.0.0.0", "--port", "8600"]
