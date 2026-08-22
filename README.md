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
