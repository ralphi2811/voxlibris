# voxlibris

**Transforme un livre en livre audio chapitré, entièrement en local.**

EPUB, PDF, scan ou texte brut en entrée. Un `.m4b` avec marqueurs de chapitres et
couverture, plus un MP3 par chapitre, en sortie. Rien ne quitte votre machine : pas
d'API, pas de compte, pas de quota — sauf si vous choisissez, explicitement, un service
distant.

> [!WARNING]
> **Le moteur par défaut, XTTS-v2, est sous licence [CPML](https://coqui.ai/cpml) :
> usage non commercial uniquement.** voxlibris ne redistribue pas ses poids — ils sont
> téléchargés sur votre machine à la première exécution, et la licence vous engage à ce
> moment-là. Pour un usage commercial, prenez Kokoro (Apache 2.0), Piper (MIT), ZONOS2
> ou OmniVoice (Apache 2.0) : c'est un choix dans la page Voix. Voir [NOTICE.md](NOTICE.md).

---

## Ce que ça fait

```
livre ──▶ ingestion ──▶ relecture ──▶ normalisation ──▶ synthèse ──▶ assemblage ──▶ .m4b
          EPUB/PDF/     (si scan)     ponctuation,      six moteurs,  chapitres,
          scan/texte                  nombres, pauses   contrôle      R128, couverture
                                                        qualité
```

Selon la source, l'effort n'est pas le même — autant le dire franchement :

| Source | Relecture nécessaire ? |
|---|---|
| **EPUB** | Aucune. Chapitres et texte sont déjà propres. |
| **EPUB issu d'un PDF** (pdf2htmlEX, pages positionnées) | Presque aucune. Lu comme un PDF : paragraphes recomposés, folios et étiquettes de dessins écartés, glyphes de ligatures retrouvés par les polices embarquées puis par le dictionnaire. |
| **PDF texte natif** | Marginale. |
| **PDF scanné** | **Oui, et c'est le gros du travail.** L'OCR se trompe, et un `1l` lu à voix haute ne pardonne pas. |

voxlibris ne supprime pas cette relecture, il la rend rapide : texte éditable face à
l'image de la page, mots suspects surlignés, écoute d'un segment en un clic, et un
modèle de langage local qui propose — sans jamais appliquer — une correction pour chaque
forme signalée.

## Démarrage

Il faut Docker avec Compose v2, une quinzaine de gigaoctets de disque pour l'image de
l'atelier, et, pour XTTS et les cloneurs de voix, une carte NVIDIA avec
[`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Sans carte, Piper et Kokoro tournent sur processeur.

```bash
git clone https://github.com/ralphi2811/voxlibris && cd voxlibris
cp .env.example .env
docker compose pull
docker compose up
```

Les images sont publiées sur ghcr.io par l'intégration continue à chaque changement :
`pull` les télécharge. Sans lui, `up` les construit sur place, ce qui prend dix minutes
pour l'atelier.

L'interface est sur `http://localhost:8000`. Trois conteneurs tournent : l'interface,
légère ; l'atelier, qui porte les moteurs de synthèse ; et un relais vers Docker, réduit
à quatre requêtes, dont il est question plus bas. Les projets vivent dans `./data` sur
l'hôte — `VOXLIBRIS_DATA_DIR` dans le `.env` pour les mettre ailleurs.

**Avec une carte NVIDIA**, empilez la surcharge qui la donne à l'atelier :

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

Elle est séparée du fichier principal pour qu'un `docker compose up` nu réussisse sur une
machine sans carte. Les serveurs de synthèse optionnels sont des **profils** Compose, à
déclarer avec cette même paire de fichiers :

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --profile omnivoice --profile zonos2 --profile voxtral pull
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --profile omnivoice --profile zonos2 --profile voxtral up -d
```

Déclarez-en autant que vous voulez : ils ne tiennent pas tous sur une carte à la fois,
et l'atelier s'en charge, voir [le concierge](#le-concierge-de-la-carte-graphique). Leurs
poids se téléchargent au premier démarrage, dans le volume `models`, qui survit aux
reconstructions.

| Image | Taille | Contenu |
|---|---|---|
| `ghcr.io/ralphi2811/voxlibris-web` | 0,5 Go | l'interface |
| `ghcr.io/ralphi2811/voxlibris-worker` | 15 Go | PyTorch, XTTS, Kokoro, Piper, Tesseract, ffmpeg |
| `ghcr.io/ralphi2811/voxlibris-omnivoice` | 13 Go | PyTorch et OmniVoice, profil `omnivoice` seulement |
| `ghcr.io/ralphi2811/voxlibris-voxtral` | 30 Go | vLLM, profil `voxtral` seulement |
| `ghcr.io/ralphi2811/voxlibris-zonos2` | 37 Go | serveur de Zyphra sur CUDA complet, profil `zonos2` seulement |

Toutes sont publiées par l'intégration continue à chaque changement qui les concerne ;
`build:` reste dans le Compose pour qui modifie le code.

**Un premier livre.** `samples/maupassant-le-horla.epub` est du domaine public et
traverse la chaîne sans intervention : déposez-le dans la Bibliothèque, choisissez une
voix, lancez la synthèse. Sept nouvelles ; la première fait une heure d'écoute.

Les conteneurs tournent sous votre UID, pour que rien de ce qu'ils écrivent dans
`./data` n'appartienne à root. **Si votre identifiant n'est pas 1000**, dites-le à
Compose — les shells ne l'exportent pas :

```bash
printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" >> .env
```

## L'interface

L'Atelier suit la chaîne du livre, une page par étape, et la barre latérale dit d'un
coup d'œil où en est chaque livre et ce qui reste à faire.

- **Bibliothèque** : dépôt par glisser-déposer, titre, auteur et langue proposés d'après
  le fichier, état de chaque livre, téléchargement.
- **Chapitres** : titres corrigés en place — ils sont annoncés à voix haute —, chapitres
  retirés ou recollés, métadonnées et couverture du livre audio. La couverture est tirée
  du livre — page de titre d'un PDF, image désignée par un EPUB —, remplaçable par une
  image à soi, ou cherchée sur Open Library d'un clic (seuls le titre et l'auteur partent
  sur le réseau, et seulement à la demande).
- **Relecture** : le texte face à la page d'origine — image du PDF, ou page de l'EPUB
  rendue avec ses polices —, les formes suspectes filtrées par cause, les propositions
  du modèle acceptées ou écartées une à une, rechercher-remplacer.
- **Préparation** : silences, annonce des chapitres, répliques repérées, et les segments
  tels qu'ils partent au moteur. Une piste est à jour quand elle dit exactement le texte de ses segments,
  avec la voix du livre. Un chapitre corrigé après coup est signalé, avec la marche à
  suivre : relancer la synthèse — qui refait d'abord la préparation, puis ne repasse au
  moteur que les segments dont le texte a changé —, et réassembler.
- **Voix** : les moteurs que l'installation offre, avec leur licence, un banc d'essai sur
  le même extrait, le dépôt d'un extrait à cloner, le coût du livre entier par moteur.
- **Synthèse** : réglages — moteur, narrateur, voix des dialogues, vitesse —, avancement
  segment par segment, bouton d'arrêt, et le tableau
  des **segments à l'oreille** — instant, cause, le passage autour, écouter, corriger,
  rejouer, valider. Rejouer un segment le recolle dans sa piste sans refaire le
  chapitre ; le valider le garde tel quel, l'oreille ayant le dernier mot.
- **Assemblage** et **Journal** des tâches.
- **Réglages** : modèle de langage, Voxtral, ZONOS2, OmniVoice, licence XTTS, carte
  graphique — enregistrés dans le dossier des données, pris en compte sans redémarrage,
  avec un bouton Tester par service. Le `.env` reste la couche de dessous.

L'**atelier** est le processus qui exécute les tâches — le service `worker` sous Compose.
Il bat toutes les cinq secondes dans `atelier.json` ; l'interface en déduit s'il est là,
sa carte graphique, ses moteurs, et l'état des serveurs qu'il garde.

## Moteurs de synthèse

| Moteur | Où il tourne | Licence des poids | Pour quoi faire |
|---|---|---|---|
| **XTTS-v2** | GPU, ~4 Go | CPML, **non commercial** | La meilleure prosodie en français. Le défaut. |
| **Kokoro-82M** | GPU ou processeur | Apache 2.0 | Rapide et très stable, une voix française. |
| **Piper** | Processeur | MIT | Quasi instantané : idéal pour régler pauses et vitesse avant la version finale. |
| **OmniVoice** | GPU, ~3 Go, profil `omnivoice` | Apache 2.0 | **Clone une voix** d'un extrait. Léger : tient à côté de XTTS. Six cents langues. |
| **ZONOS2** | GPU, ~21 Go, profil `zonos2` | Apache 2.0 | **Clone une voix** lui aussi ; réclame la carte pour lui seul. Quarante langues. |
| **Voxtral TTS** | API distante payante, ou GPU 16 Go sous profil `voxtral` | CC BY-NC 4.0 | Neuf langues, trente voix. Sans carte graphique en API. |

Les trois premiers vivent dans l'atelier ; les trois autres sont des serveurs à part,
sous profil, que l'atelier réveille à la demande.

### Plusieurs voix : la distribution

Un livre peut être lu à plusieurs voix, sur le même moteur. Le narrateur lit tout, sauf
ce que la **distribution** — sur la page Synthèse — confie à d'autres voix du catalogue
du moteur, une voix clonée comprise :

- **Répliques** : les paragraphes qui ouvrent sur un tiret ou des guillemets, reconnus
  à leur typographie, sans rien deviner de plus. Une réplique nichée dans un paragraphe
  de récit reste au narrateur, et l'incise « dit-elle » suit sa réplique.
- **Un persona** : les paragraphes que vous lui attribuez dans la Relecture — les
  lettres d'un frère, le journal d'un personnage. Sélectionnez la plage, nommez le
  persona, Attribuer : le texte reçoit une ligne `@Charles` avant et une ligne `@` après,
  et c'est le texte qui porte l'attribution, donc elle survit à toute correction. Un
  persona nommé sans voix est lu par le narrateur, et la synthèse le dit.

Un champ vide rend au narrateur. Changer une voix ne refait que ce qu'elle disait : le
manifeste de chaque piste note qui a dit chaque segment, le reste est repris tel quel.

### Cloner une voix : OmniVoice et ZONOS2

Un extrait de quinze à trente secondes, déposé depuis la page Voix, devient une voix pour
tous les livres — sans rien entraîner, sans transcription à fournir. Le même extrait
sert aux deux moteurs. Il doit être **dans la langue du livre**, sans quoi la lecture
prend son accent, et ce doit être **une voix dont vous avez le droit de vous servir** :
la vôtre, celle d'une personne qui y consent, une voix libre. Rien dans les licences ne
le borne, c'est vous que cela engage ; voir [NOTICE.md](NOTICE.md).

[OmniVoice](https://github.com/k2-fsa/OmniVoice), de k2-fsa, est le léger : 0,8 milliard
de paramètres, trois gigaoctets de carte en lecture, six cents langues, et en français un
taux d'erreur de mots un peu meilleur que ZONOS2. Sa sortie est à 24 kHz. Le paquet ne
livre pas de serveur ; celui de voxlibris, `docker/omnivoice-server.py`, tient en trois
routes. Whisper transcrit chaque extrait une fois, au démarrage ou au dépôt ; il prend
dix gigaoctets le temps de le faire, et passe de lui-même sur le processeur sous 12 Go de
carte. D'un extrait long, le serveur ne retient que les douze premières secondes,
coupées au silence le plus net : au-delà, le modèle clone moins bien.

[ZONOS2](https://github.com/Zyphra/Zonos2), de Zyphra, est le lourd : 16 Go de poids,
21 Go de carte, une sortie à 44,1 kHz, le français en deuxième rang derrière l'anglais,
le mandarin et le japonais. Il ne cohabite avec aucun autre serveur sur une carte de
24 Go. Il met au premier démarrage trois voix anglaises dans `data/voices`.

Sur les deux, le réglage de vitesse agit, et le contrôle qualité par segment rattrape les
rares hallucinations que leurs auteurs reconnaissent.

### Voxtral, chez Mistral ou chez vous

**En API, Voxtral est le seul moteur distant** : le texte du livre est envoyé à Mistral,
page après page. Il ne démarre pas sans `VOXLIBRIS_MISTRAL_API_KEY`, et n'est jamais
choisi par défaut. Comptez 0,016 $ pour mille caractères — environ 1,30 $ pour un
roman — que l'interface annonce avant de lancer la synthèse. Trente voix, dont six
françaises — « Marie », en six émotions. L'API n'offre aucun réglage de débit : le
réglage de vitesse y est sans effet, et le journal le dit plutôt que de l'ignorer.

Les poids sont publics, et le serveur de vLLM parle le même protocole : **seule l'adresse
change**. Le profil `voxtral` le lance sous Compose ; il pèse 16 Go de mémoire vidéo et
demande un jeton Hugging Face (`HF_TOKEN` dans le `.env`) dont le compte a accepté la
licence du dépôt. Trois choses disparaissent du même coup : la facture, le filtre de
modération — celui de l'API refuse des passages parfaitement littéraires — et l'envoi du
texte à un tiers. En contrepartie, servi chez vous, Voxtral **ne clone pas** : les poids
ouverts n'embarquent pas l'encodeur audio que le clonage réclame. Vingt et une voix
fournies, dont `fr_female` et `fr_male`, et le réglage de vitesse y agit.

Pour un vLLM lancé à la main sur la machine plutôt que sous Compose :

```bash
uv venv ~/.local/share/voxlibris-vllm/.venv
uv pip install --python ~/.local/share/voxlibris-vllm/.venv/bin/python "vllm>=0.28" "vllm-omni>=0.28" ninja
PATH="$HOME/.local/share/voxlibris-vllm/.venv/bin:$PATH" \
  ~/.local/share/voxlibris-vllm/.venv/bin/vllm serve mistralai/Voxtral-4B-TTS-2603 \
  --omni --port 8600 --gpu-memory-utilization 0.80
```

puis `VOXLIBRIS_MISTRAL_BASE_URL=http://host.docker.internal:8600/v1` dans les Réglages.
`ninja` n'est pas facultatif : sans lui, vLLM compile un noyau de tri au démarrage et
échoue sur un `FileNotFoundError: 'ninja'` enfoui à cinquante lignes d'une trace qui
parle d'échec d'initialisation du moteur. Il lui faut aussi le CUDA Toolkit, pour `nvcc`.
**vLLM s'installe dans son propre environnement** — sa version de PyTorch se querellerait
avec celle de XTTS.

### Le concierge de la carte graphique

Les serveurs sous profil ne tiennent pas tous sur une carte à la fois, et ce n'est pas
grave : l'atelier **réveille le serveur qu'une tâche demande, endort ceux qui ne
tiendraient pas à côté**, et rend la carte après un quart d'heure sans rien faire —
réglable dans les Réglages, zéro pour jamais, avec un bouton pour la rendre tout de
suite. Un serveur en veille reste proposé sur la page Voix ; le réveil se paie au début
de la tâche — une vingtaine de secondes pour ZONOS2, une douzaine pour OmniVoice, deux
minutes pour Voxtral, plus le téléchargement des poids la toute première fois — et se
lit dans la progression. Les adresses des services Compose vont alors de soi, rien à
saisir dans les Réglages. Un serveur perdu en pleine tâche est réveillé et le chapitre
repris.

Pour cela, l'atelier parle à Docker par le service `docker-proxy` du Compose, un relais
HAProxy dont les règles, `docker/docker-proxy.cfg`, n'ouvrent que quatre requêtes —
lister les conteneurs du projet, en inspecter un, le démarrer, l'arrêter — là où la
socket brute vaudrait root sur la machine. Sans lui, `VOXLIBRIS_DOCKER_URL` vide, rien
n'est réveillé ni endormi : l'atelier voit ce qui répond, comme un serveur lancé à la
main, et les adresses se renseignent dans les Réglages.

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
au segment qui l'a produit, et un tampon `chNN.voice` qui dit quelle voix l'a lu. Entendez
quelque chose à 12:34, retrouvez le segment fautif ; changez de voix, seules les pistes
lues par l'ancienne sont refaites.

## Assistance par modèle de langage

Un modèle local peut prendre en charge la partie fastidieuse de la relecture d'un scan. Le
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
aurait fait perdre à un personnage sa voix.

Mesurez-le avant de lui faire confiance : si vous disposez d'un texte océrisé **et** de sa
relecture manuelle, `voxlibris bench-proofread text/raw text/clean` compare les deux et
compte les fautes corrigées, les fautes manquées et — la seule ligne qui décide — les
modifications proposées sur un texte qui était déjà juste.

N'importe quel service parlant le protocole OpenAI convient, d'Ollama à llama.cpp, vLLM
ou un fournisseur distant :

```bash
VOXLIBRIS_LLM_BASE_URL=http://host.docker.internal:11434/v1
VOXLIBRIS_LLM_MODEL=gemma4:12b
```

L'étape est **facultative** : sans modèle joignable, la relecture se fait à la main.
L'adresse par défaut est locale, donc aucun texte ne quitte la machine tant que vous
n'avez pas vous-même désigné un service distant.

## En ligne de commande

L'interface web n'est qu'une façade sur les mêmes fonctions. Sous Compose, la ligne de
commande passe par l'atelier, et les fichiers doivent être sous `./data`, qui est
`/data` dans le conteneur :

```bash
docker compose run --rm worker voxlibris ingest /data/livre.epub /data/projet
docker compose run --rm worker voxlibris synth /data/projet --backend kokoro
docker compose run --rm worker voxlibris assemble /data/projet
```

`synth --dialogue-voice "Ana Florence"` confie les répliques à une seconde voix du même
moteur, `--cast "Charles=Damien"` donne sa voix à un persona du texte ; `-` ou une voix
vide rendent au narrateur.

La ligne de commande vise la carte par défaut : sans carte, ajoutez `--device cpu` à
`synth`. `voxlibris run livre.epub projet/` enchaîne le tout, et refuse de le faire pour un
scan, qui demande une relecture. `voxlibris review`, `proofread`, `normalize`, `voices`,
`status`, `drop-chapter` et `merge-chapter` couvrent le reste ; `voxlibris --help` les
détaille.

## Développement

```bash
uv sync --group dev          # le cœur, sans PyTorch ni moteur
uv run pytest -q             # 294 tests, quelques secondes, aucune carte requise
uv run ruff check . && uv run ruff format --check .
```

Les moteurs s'installent avec `uv sync --extra tts --extra ocr`, ce qui tire PyTorch.
L'intégration continue de GitHub fait exactement les trois lignes ci-dessus, sur
Ubuntu, sans carte.

Après une modification de `src/`, reconstruisez l'image concernée avec la même paire de
fichiers Compose que celle qui a démarré l'atelier — sans la surcharge, le worker repart
sans sa carte. Les Dockerfiles installent les dépendances avant de copier le code : la
reconstruction ne refait pas PyTorch, et l'intégration continue, qui publie les images
sur ghcr.io avec son cache de couches, non plus.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build --no-deps worker web
```

## Dépannage

- **« localhost » depuis un conteneur** désigne le conteneur lui-même. Pour joindre un
  Ollama ou un vLLM installé sur la machine, l'adresse est `http://host.docker.internal:…`.
- **Un pare-feu sur l'hôte bloque ces adresses** : `ufw`, en particulier, rejette ce qui
  arrive des ponts Docker, et le conteneur voit un « timed out » plutôt qu'un refus.
  Ouvrez le port aux réseaux Docker, et à eux seuls :
  `sudo ufw allow from 172.16.0.0/12 to any port 11434 proto tcp`. Les services sous
  profil ne posent pas ce problème : les conteneurs se parlent sans passer par le
  pare-feu de l'hôte.
- **« CUDA unknown error » dans les conteneurs après un redémarrage**, alors que
  `nvidia-smi` va bien sur l'hôte : la spécification CDI du toolkit a été générée avant
  le chargement du module `nvidia_uvm` et pointe un mauvais périphérique. Régénérez-la
  puis recréez les conteneurs :

  ```bash
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  echo nvidia_uvm | sudo tee /etc/modules-load.d/nvidia-uvm.conf
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate
  ```

- **Un serveur sous profil ne se réveille pas** : `docker logs voxlibris-zonos2-1` (ou
  `-omnivoice-1`, `-voxtral-1`) dit pourquoi ; le premier démarrage télécharge les poids
  et peut prendre plusieurs minutes.

## État du projet

voxlibris a été développé et éprouvé sur une seule machine — Linux, une carte de 24 Go —, sur
des livres en français : c'est la langue de sa relecture, de sa normalisation et de ses
voix par défaut. Les autres langues des moteurs sont accessibles mais peu essayées. La
détection de structure des PDF scannés est une heuristique calibrée sur quelques
ouvrages ; l'interface est là pour rattraper ce qu'elle manque. Les retours, avec le
format du livre et le moteur en cause, sont bienvenus dans les
[issues](https://github.com/ralphi2811/voxlibris/issues).

## Licence

Code sous [MIT](LICENSE). Les modèles de synthèse ont leurs propres licences, détaillées
dans [NOTICE.md](NOTICE.md) — **lisez-le avant un usage commercial**.

voxlibris ne fournit aucun livre. Convertir une œuvre encore protégée pour votre usage
personnel relève, selon les pays, de la copie privée ; la diffuser, non.
