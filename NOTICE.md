# Licences des composants tiers

Le code de voxlibris est sous licence MIT (voir `LICENSE`). Les **modèles de synthèse
vocale** qu'il pilote ont leurs propres licences, distinctes, que voxlibris ne peut pas
étendre ni assouplir.

voxlibris **ne redistribue aucun poids de modèle**. Ils sont téléchargés à l'exécution,
sur la machine de l'utilisateur, depuis leurs dépôts respectifs. La licence du modèle
engage donc l'utilisateur au moment où il choisit de l'exécuter.

## Moteurs de synthèse

| Moteur | Licence des poids | Portée |
|---|---|---|
| **XTTS-v2** (Coqui) | [CPML](https://coqui.ai/cpml) | **Usage non commercial uniquement** |
| **Kokoro-82M** | Apache 2.0 | Aucune restriction d'usage |
| **Piper** | MIT | Aucune restriction d'usage |
| **Voxtral TTS** (Mistral) | CC BY-NC 4.0 | **Poids non commerciaux ; l'API, payante, l'autorise** |
| **ZONOS2** (Zyphra) | Apache 2.0 | Aucune restriction d'usage ; serveur sous MIT |
| **OmniVoice** (k2-fsa) | Apache 2.0 | Aucune restriction d'usage ; Whisper (MIT) transcrit les extraits |

### Voxtral TTS : la restriction est à l'envers des autres

Pour les trois premiers, voxlibris exécute des poids téléchargés sur votre machine. Pour
Voxtral, il appelle l'API de Mistral, et **le texte du livre quitte donc la machine** :
c'est le seul moteur dans ce cas, et c'est à ce titre qu'il n'est jamais choisi par
défaut. Il ne démarre pas sans `VOXLIBRIS_MISTRAL_API_KEY`.

La curiosité de ce modèle est que l'usage commercial passe par la voie payante, et non
par les poids ouverts — l'inverse de XTTS, dont les poids sont gratuits mais restreints.
Les conditions d'utilisation de l'API de Mistral s'appliquent, y compris son interdiction
de cloner une voix sans autorisation.

Pour un livre encore protégé, l'envoyer à un tiers n'est pas anodin : c'est une
transmission, là où la synthèse locale reste une copie privée.

### ZONOS2 et OmniVoice : le clonage engage celui qui dépose l'extrait

ZONOS2 et OmniVoice clonent une voix à partir de quelques secondes d'audio, sans rien
demander d'autre. Leurs poids sont sous Apache 2.0, le serveur de Zyphra sous MIT, celui
d'OmniVoice est le nôtre — rien, dans les licences, ne borne cet usage. C'est donc à vous de vous en tenir à des voix dont vous avez le droit
de vous servir : la vôtre, celle d'une personne qui y consent, une voix libre de droits.
Cloner quelqu'un à son insu relève du droit à la voix et à l'image, indépendamment de
toute licence logicielle. Les extraits restent dans `data/voices`, sur votre machine.

### ⚠️ XTTS-v2 est le moteur par défaut

Il offre la meilleure qualité en français, et c'est pourquoi il est proposé par défaut.
Mais la **Coqui Public Model License interdit tout usage commercial**.

Si votre usage est commercial, changez de moteur — c'est une ligne de configuration :

```yaml
# config.yaml
tts:
  backend: kokoro   # Apache 2.0, ou "piper" pour du MIT
```

La première exécution de XTTS suppose l'acceptation de la CPML, matérialisée par la
variable d'environnement `COQUI_TOS_AGREED=1`. Elle n'est **pas** positionnée par défaut :
c'est un acte délibéré de l'utilisateur.

## Modèle de langage pour la relecture

voxlibris n'embarque ni ne recommande aucun modèle de langage : il se contente d'appeler
un service compatible avec le protocole OpenAI, dont l'adresse et le nom de modèle sont
donnés dans le `.env`. Le choix du modèle — et donc de sa licence — appartient
entièrement à l'utilisateur.

À titre d'exemple, les modèles publiés sous **Apache 2.0** (Qwen, Mistral) n'imposent
aucune restriction d'usage, là où d'autres familles assortissent leur diffusion de
conditions particulières. Vérifiez celle du modèle que vous configurez.

Cette étape est facultative et désactivable (`VOXLIBRIS_LLM_ENABLED=0`). L'adresse par
défaut est locale : **aucun texte ne quitte la machine** tant que l'utilisateur n'a pas
lui-même désigné un service distant — ce qui, pour un ouvrage protégé, mérite réflexion.

## Autres dépendances

- **PyMuPDF** — AGPL-3.0. Utilisée comme bibliothèque pour lire les PDF. Si vous
  distribuez un service en réseau bâti sur voxlibris, cette licence a des implications :
  vérifiez-les. Une alternative sous licence permissive (`pypdfium2`, BSD) est envisagée.
- **Tesseract OCR** — Apache 2.0.
- **FFmpeg** — LGPL/GPL selon la compilation, invoqué comme programme externe.
- Le reste de la pile Python (FastAPI, ebooklib, num2words…) est sous licences
  permissives, détaillées dans les métadonnées de chaque paquet.

## Contenu traité

voxlibris ne fournit aucun livre. Les fichiers du dossier `samples/` proviennent du
**domaine public**. Les livres que vous traitez restent votre affaire : convertir en
audio une œuvre encore protégée pour votre usage personnel relève, selon les pays, de la
copie privée — la diffuser, non.
