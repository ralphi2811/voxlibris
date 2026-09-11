# voxlibris

**Transforme un livre en livre audio chapitré, entièrement en local.**

EPUB, PDF, scan, texte brut en entrée. Un `.m4b` avec marqueurs de chapitres et
couverture, plus un MP3 par chapitre, en sortie. Rien ne quitte votre machine : pas
d'API, pas de compte, pas de quota.

> [!WARNING]
> **Le moteur par défaut, XTTS-v2, est sous licence [CPML](https://coqui.ai/cpml) :
> usage non commercial uniquement.** voxlibris ne redistribue pas ses poids — ils sont
> téléchargés sur votre machine à la première exécution, et la licence vous engage à ce
> moment-là. Pour un usage commercial, basculez sur Kokoro (Apache 2.0) ou Piper (MIT) :
> c'est une ligne de configuration. Voir [NOTICE.md](NOTICE.md).

---

## Ce que ça fait

```
livre ──▶ ingestion ──▶ relecture ──▶ normalisation ──▶ synthèse ──▶ assemblage ──▶ .m4b
          EPUB/PDF/     (si scan)     ponctuation,      XTTS/Kokoro/  chapitres,
          scan/texte                  nombres, pauses   Piper         R128, couverture
```

Selon la source, l'effort n'est pas le même — autant le dire franchement :

| Source | Relecture nécessaire ? |
|---|---|
| **EPUB** | Aucune. Chapitres et texte sont déjà propres. |
| **EPUB issu d'un PDF** (pdf2htmlEX, pages positionnées) | Presque aucune. Lu comme un PDF : paragraphes recomposés, folios et étiquettes de dessins écartés, glyphes de ligatures retrouvés par les polices embarquées puis par le dictionnaire. |
| **PDF texte natif** | Marginale. |
| **PDF scanné** | **Oui, et c'est le gros du travail.** L'OCR se trompe, et un `1l` lu à voix haute ne pardonne pas. |

voxlibris ne supprime pas cette relecture, il la rend rapide : texte éditable face à
l'image de la page, mots suspects surlignés, écoute d'un segment en un clic.

### Assistance par modèle de langage

Un modèle local peut prendre en charge la partie fastidieuse de cette relecture. Le
cadrage est strict, et c'est ce qui le rend utilisable :

- il ne voit que les formes **déjà signalées** comme suspectes, une par une, jamais le
  texte entier — il lui est donc impossible de réécrire ce qu'on ne lui a pas montré ;
- ce qu'il propose est borné : distance d'édition limitée, deux mots au maximum, aucune
  ponctuation ajoutée, et le remplacement doit exister au dictionnaire ;
- une forme qui **revient dans l'ouvrage** ne lui est pas soumise : ce qui se répète est
  voulu — le parler d'un personnage, un néologisme — et non une coquille ;
- **rien n'est appliqué.** Les propositions s'affichent dans l'éditeur, avec leur
  contexte, et vous les acceptez ou les écartez d'un clic.

Cette prudence n'est pas de principe. Sur notre livre de référence, le modèle a proposé
de corriger « Mâdâme » en « Madame » — la graphie est celle d'un perroquet, et l'accepter
aurait fait perdre à un personnage sa voix. Le contexte affiché à côté de chaque
proposition sert exactement à cela.

Mesurez-le avant de lui faire confiance : si vous disposez d'un texte océrisé **et** de sa
relecture manuelle, `voxlibris bench-proofread text/raw text/clean` compare les deux et
compte les fautes corrigées, les fautes manquées et — la seule ligne qui décide — les
modifications proposées sur un texte qui était déjà juste.

La configuration tient dans le `.env` : n'importe quel service parlant le protocole
OpenAI convient, d'Ollama à llama.cpp, vLLM ou un fournisseur distant.

```bash
VOXLIBRIS_LLM_BASE_URL=http://localhost:11434/v1
VOXLIBRIS_LLM_MODEL=gemma4:12b
```

L'étape est **facultative** : sans modèle joignable, la relecture se fait à la main comme
avant. L'adresse par défaut est locale, donc aucun texte ne quitte la machine tant que
vous n'avez pas vous-même désigné un service distant.

## Qualité de la synthèse

Un modèle autorégressif comme XTTS peut, sur un segment isolé, boucler ou partir dans
une autre langue — sans lever la moindre erreur. voxlibris mesure donc chaque segment
produit et rejoue ceux qui sortent des clous :

- **durée rapportée au texte**, avec un débit calibré automatiquement sur la voix choisie ;
- **plafond absolu**, qui rattrape les phrases courtes que le contrôle par ratio laisse
  passer ;
- **débordement de la dernière phrase**, mesuré séparément — c'est la signature du babil
  ajouté en fin d'énoncé, invisible sur la durée totale ;
- **repli par découpe** : si un segment résiste, il est rejoué phrase par phrase.

Chaque chapitre produit un manifeste `chNN.timing.json` reliant chaque instant du fichier
au segment qui l'a produit. Entendez quelque chose à 12:34, retrouvez le segment fautif.

## Moteurs de synthèse

| Moteur | Où il tourne | Licence des poids | Pour quoi faire |
|---|---|---|---|
| **XTTS-v2** | GPU, ~4 Go | CPML, **non commercial** | La meilleure prosodie en français. Le défaut. |
| **Kokoro-82M** | GPU ou processeur | Apache 2.0 | Rapide et très stable, une voix française. |
| **Piper** | Processeur | MIT | Quasi instantané : idéal pour régler pauses et vitesse avant la version finale. |
| **Voxtral TTS** | **API distante**, payante | CC BY-NC 4.0 | Neuf langues, sans carte graphique. |
| **ZONOS2** | GPU, ~16 Go, serveur sous profil | Apache 2.0 | **Clone une voix** d'un simple extrait. Quarante langues. |

Trente voix sont fournies, dont **six françaises** — « Marie », en six émotions, de
`Neutral` à `Curious`. `voxlibris voices voxtral` liste celles que voit votre compte, et
vous pouvez y ajouter une voix clonée depuis la console de Mistral, à partir d'un
échantillon dont vous avez le droit de vous servir : leurs conditions interdisent de
cloner quelqu'un sans son accord.

### Voxtral chez vous plutôt que chez Mistral

Les poids sont publics, et le serveur de vLLM parle le même protocole que l'API : **seule
l'adresse change**, le code de voxlibris est le même. Deux différences tout de même :
le serveur local **sait moduler le débit**, le réglage de vitesse y agit ; et il **ne
sait pas cloner une voix** — les poids ouverts n'embarquent pas l'encodeur audio que le
clonage réclame, et une demande de ce genre fait tomber le moteur. Vingt et une voix
fournies, dont deux françaises.

```bash
uv venv ~/.local/share/voxlibris-vllm/.venv
uv pip install --python ~/.local/share/voxlibris-vllm/.venv/bin/python "vllm>=0.28" "vllm-omni>=0.28" ninja
PATH="$HOME/.local/share/voxlibris-vllm/.venv/bin:$PATH" \
  ~/.local/share/voxlibris-vllm/.venv/bin/vllm serve mistralai/Voxtral-4B-TTS-2603 \
  --omni --port 8600 --gpu-memory-utilization 0.80
```

`ninja` n'est pas facultatif : sans lui, vLLM compile un noyau de tri au démarrage et
échoue sur un `FileNotFoundError: 'ninja'` enfoui à cinquante lignes d'une trace d'appels
qui parle d'échec d'initialisation du moteur. Il lui faut aussi le CUDA Toolkit, pour
`nvcc`.

Puis dans le `.env` : `VOXLIBRIS_MISTRAL_BASE_URL=http://localhost:8600/v1`, et plus
besoin de clé.

Comptez 8 Go de poids et 16 Go de mémoire vidéo. Le dépôt est sous conditions : il faut
accepter la licence avec son compte Hugging Face. **vLLM s'installe dans son propre
environnement** — sa version de PyTorch se querellerait avec celle de XTTS.

Trois choses disparaissent du même coup : la facture, le filtre de modération — celui de
l'API refuse des passages parfaitement littéraires, voir plus bas — et l'envoi du texte à
un tiers. Les voix ne sont plus le catalogue du compte mais les plongements livrés avec
les poids, dont `fr_female` et `fr_male`.

### ZONOS2 : cloner une voix, chez vous

[ZONOS2](https://github.com/Zyphra/Zonos2), de Zyphra, est le seul moteur de voxlibris qui
**clone une voix**, et il le fait sans rien entraîner : un extrait de dix à trente
secondes, déposé depuis la page Voix, devient une voix pour tous les livres. Pas de
transcription à fournir. Les poids sont sous Apache 2.0, le serveur sous MIT — et
l'extrait, lui, doit être une voix dont vous avez le droit de vous servir, voir
`NOTICE.md`.

C'est un service à part, sous profil, comme Voxtral local :

```bash
docker compose --profile zonos2 -f docker-compose.yml -f docker-compose.gpu.yml up
```

puis `VOXLIBRIS_ZONOS2_BASE_URL=http://zonos2:1919` dans le `.env` ou les Réglages. Les
poids (16 Go, dépôt ouvert, sans jeton) se téléchargent au premier démarrage dans le
volume `models`. Comptez une carte de 24 Go : ZONOS2 et Voxtral local ne tiennent pas
ensemble sur une seule, lancez l'un ou l'autre. Les voix sont les fichiers audio de
`data/voices` ; le serveur relit ce dossier à chaque demande, une voix déposée existe
aussitôt. Il y met au premier démarrage les trois voix anglaises livrées avec le dépôt.

Le réglage de vitesse agit, la langue de normalisation du texte suit celle du livre
(neuf langues, dont le français), et le contrôle qualité par segment rattrape les rares
hallucinations que le rapport technique de Zyphra reconnaît. Le français y est en
deuxième rang, derrière l'anglais, le mandarin et le japonais.

**Voxtral en API est le seul moteur distant** : le texte du livre est envoyé à Mistral,
page après page. Il ne démarre pas sans `VOXLIBRIS_MISTRAL_API_KEY`, et n'est jamais choisi
par défaut. Comptez 0,016 $ pour mille caractères — environ 1,30 $ pour un roman — que
l'interface annonce avant de lancer la synthèse. L'API n'offre aucun réglage de débit :
le réglage de vitesse y est sans effet, et le journal le dit plutôt que de l'ignorer ;
servi chez vous, le même moteur l'applique.

## L'interface

L'Atelier suit la chaîne du livre, une page par étape, et la barre latérale dit d'un
coup d'œil où en est chaque livre et ce qui reste à faire.

- **Bibliothèque** : dépôt par glisser-déposer, titre, auteur et langue proposés d'après
  le fichier, état de chaque livre, téléchargement.
- **Chapitres** : titres corrigés en place — ils sont annoncés à voix haute —, chapitres
  retirés ou recollés, métadonnées et couverture du livre audio. La
  couverture est tirée du livre — page de titre d'un PDF, image désignée par un EPUB —,
  remplaçable par une image à soi, ou cherchée sur Open Library d'un clic (seuls le titre
  et l'auteur partent sur le réseau, et seulement à la demande).
- **Relecture** : le texte face à la page d'origine — image du PDF, ou page de l'EPUB
  rendue avec ses polices —, les formes suspectes filtrées par cause, les propositions
  du modèle acceptées ou écartées une à une, rechercher-remplacer.
- **Préparation** : silences, annonce des chapitres, et les segments tels qu'ils partent
  au moteur. Une piste est à jour quand elle dit exactement le texte de ses segments.
  Un chapitre corrigé après coup est signalé, avec la marche à suivre : relancer la
  synthèse — qui refait d'abord la préparation, puis ne repasse au moteur que les
  segments dont le texte a changé —, et réassembler.
- **Voix** : les moteurs avec leur licence et ce que l'atelier sait charger, un banc
  d'essai sur le même extrait, le coût du livre entier par moteur.
- **Synthèse** : réglages, avancement segment par segment, bouton d'arrêt, et le tableau
  des **segments à l'oreille** — instant, cause, le passage autour, écouter, corriger,
  rejouer, valider. Rejouer un segment le recolle dans sa piste sans refaire le
  chapitre ; le valider le garde tel quel, l'oreille ayant le dernier mot.
- **Assemblage** et **Journal** des tâches.
- **Réglages** : modèle de langage, Voxtral, ZONOS2, licence XTTS, matériel — enregistrés dans le
  dossier des données, partagés avec l'atelier, pris en compte sans redémarrage, avec un
  bouton Tester par service. Le `.env` reste la couche de dessous.

L'**atelier** est le processus qui exécute les tâches — le service `worker` sous Compose.
Il bat toutes les cinq secondes dans `atelier.json` ; l'interface en déduit s'il est là,
sa carte graphique et ses moteurs.

## Démarrage

```bash
git clone https://github.com/ralphi2811/voxlibris && cd voxlibris
cp .env.example .env
docker compose up
```

L'interface est sur `http://localhost:8000`. Deux conteneurs tournent : l'interface,
légère, et l'atelier, qui porte les moteurs de synthèse. Les projets vivent dans `./data`
sur l'hôte — `VOXLIBRIS_DATA_DIR` dans le `.env` pour les mettre ailleurs.

**Sans GPU**, Piper et Kokoro tournent sur processeur, et c'est ce que fait la commande
ci-dessus. **Avec une carte NVIDIA** et `nvidia-container-toolkit`, empilez la surcharge
qui la donne à l'atelier — XTTS en a besoin :

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

Le serveur Voxtral local est un troisième service, sous profil, parce qu'il pèse 16 Go de
mémoire vidéo et demande un jeton Hugging Face (`HF_TOKEN` dans le `.env`) :

```bash
docker compose --profile voxtral -f docker-compose.yml -f docker-compose.gpu.yml up
```

puis `VOXLIBRIS_MISTRAL_BASE_URL=http://voxtral:8600/v1` dans le `.env`. ZONOS2 suit le
même modèle avec `--profile zonos2` et `VOXLIBRIS_ZONOS2_BASE_URL=http://zonos2:1919`.
Pour joindre
un Ollama installé sur la machine depuis les conteneurs, l'adresse est
`http://host.docker.internal:11434/v1` — de même, `…:8600/v1` pour un vLLM lancé à la
main sur l'hôte.

**Un pare-feu sur l'hôte bloque ces adresses** : `ufw`, en particulier, rejette ce qui
arrive des ponts Docker, et le conteneur voit un « timed out » plutôt qu'un refus. Il
faut ouvrir le port aux réseaux Docker, et à eux seuls :

```bash
sudo ufw allow from 172.16.0.0/12 to any port 8600 proto tcp comment 'voxlibris : vLLM depuis Docker'
```

Le service `voxtral` sous profil ne pose pas ce problème : les conteneurs se parlent
entre eux sans passer par le pare-feu de l'hôte.

Les poids des modèles vont dans un volume nommé, `models`, et survivent aux
reconstructions d'images. Les conteneurs tournent sous votre UID : rien de ce qu'ils
écrivent dans `./data` n'appartient à root. **Si votre identifiant n'est pas 1000**,
dites-le à Compose — les shells ne l'exportent pas :

```bash
printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" >> .env
```

Le premier `up` construit les images, et l'atelier est lourd — c'est PyTorch avec CUDA :

| Image | Taille | Contenu |
|---|---|---|
| `voxlibris-web` | 0,5 Go | l'interface |
| `voxlibris-worker` | 15 Go | PyTorch, les moteurs, Tesseract, ffmpeg |
| `voxlibris-voxtral` | 30 Go | vLLM, profil `voxtral` seulement |
| `voxlibris-zonos2` | 37 Go | serveur de Zyphra sur CUDA complet, profil `zonos2` seulement |

Les poids eux-mêmes se téléchargent à la première synthèse, dans le volume `models`.

La ligne de commande passe par le même atelier ; les fichiers doivent être sous `./data`,
qui est `/data` dans le conteneur :

```bash
docker compose run --rm worker voxlibris ingest /data/livre.epub --out /data/projet
```

### En ligne de commande

L'interface web n'est pas obligatoire, tout est accessible en CLI :

```bash
voxlibris ingest livre.epub --out projet/
voxlibris synth projet/ --backend xtts --voice "Viktor Menelaos"
voxlibris assemble projet/ --title "Mon livre" --author "Un auteur"
```

## Licence

Code sous [MIT](LICENSE). Les modèles de synthèse ont leurs propres licences, détaillées
dans [NOTICE.md](NOTICE.md) — **lisez-le avant un usage commercial**.

voxlibris ne fournit aucun livre. Convertir une œuvre encore protégée pour votre usage
personnel relève, selon les pays, de la copie privée ; la diffuser, non.
