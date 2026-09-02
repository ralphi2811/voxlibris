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

Trente voix sont fournies, dont **six françaises** — « Marie », en six émotions, de
`Neutral` à `Curious`. `voxlibris voices voxtral` liste celles que voit votre compte, et
vous pouvez y ajouter une voix clonée depuis la console de Mistral, à partir d'un
échantillon dont vous avez le droit de vous servir : leurs conditions interdisent de
cloner quelqu'un sans son accord.

### Voxtral chez vous plutôt que chez Mistral

Les poids sont publics, et le serveur de vLLM parle le même protocole que l'API : **seule
l'adresse change**, le code de voxlibris est le même.

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

**Voxtral en API est le seul moteur distant** : le texte du livre est envoyé à Mistral,
page après page. Il ne démarre pas sans `VOXLIBRIS_MISTRAL_API_KEY`, et n'est jamais choisi
par défaut. Comptez 0,016 $ pour mille caractères — environ 1,30 $ pour un roman — que
l'interface annonce avant de lancer la synthèse. L'API n'offre aucun réglage de débit :
le réglage de vitesse y est sans effet, et le journal le dit plutôt que de l'ignorer.

## Démarrage

```bash
git clone https://github.com/ralphi2811/voxlibris && cd voxlibris
cp .env.example .env
docker compose up
```

L'interface est sur `http://localhost:8000`.

Le GPU est optionnel : sans lui, Piper et Kokoro tournent sur processeur. XTTS demande
une carte NVIDIA (~4 Go de VRAM) et le paquet `nvidia-container-toolkit`.

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
