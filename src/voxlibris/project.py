"""Un projet : un livre en cours de transformation, et l'état où il en est.

Toute l'arborescence et les métadonnées sont réunies ici, de sorte que la ligne de
commande et l'interface web manipulent la même chose. Les étapes sont volontairement
matérialisées par des fichiers plutôt que par un état en mémoire : la synthèse dure des
dizaines de minutes, et il faut pouvoir l'interrompre, la reprendre, ou ne refaire qu'un
chapitre sans repartir de zéro.

    projet/
    ├── project.json          métadonnées et voix retenue
    ├── text/raw/chNN.md      extraction brute
    ├── text/clean/chNN.md    après relecture — source de vérité
    ├── work/segments/*.jsonl segments prêts pour la synthèse
    └── out/wav, out/mp3, *.m4b
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .assemble import BookMetadata
from .document import Document


@dataclass
class Project:
    root: Path
    title: str = ""
    author: str = "Inconnu"
    year: str | None = None
    language: str = "fr"
    source: str | None = None
    kind: str | None = None
    needs_review: bool = False
    backend: str | None = None
    voice: str | None = None
    # Débit de parole, et étirement de tous les silences. Les moteurs sont réglés pour
    # la phrase de démonstration, pas pour une heure d'écoute : à l'oreille, la lecture
    # court et la ponctuation s'efface. Ces deux réglages sont indépendants — la vitesse
    # oblige à resynthétiser, les pauses seulement à repréparer les segments.
    speed: float = 1.0
    pause_scale: float = 1.0
    notes: dict = field(default_factory=dict)

    # --- Arborescence ---------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.root / "text" / "raw"

    @property
    def clean_dir(self) -> Path:
        return self.root / "text" / "clean"

    @property
    def segments_dir(self) -> Path:
        return self.root / "work" / "segments"

    @property
    def wav_dir(self) -> Path:
        return self.root / "out" / "wav"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def calibration_file(self) -> Path:
        return self.root / "work" / "voices.json"

    @property
    def suggestions_file(self) -> Path:
        """Propositions de correction du modèle, en attente d'un arbitrage humain."""
        return self.root / "work" / "suggestions.json"

    @property
    def config_file(self) -> Path:
        return self.root / "project.json"

    @property
    def text_dir(self) -> Path:
        """Le texte faisant foi : la version relue si elle existe, la brute sinon."""
        return self.clean_dir if any(self.clean_dir.glob("ch*.md")) else self.raw_dir

    @property
    def metadata(self) -> BookMetadata:
        return BookMetadata(
            title=self.title or self.root.name,
            author=self.author,
            year=self.year,
            narrator=f"{self.backend}/{self.voice}" if self.voice else None,
        )

    # --- Persistance ----------------------------------------------------------------
    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "title": self.title,
            "author": self.author,
            "year": self.year,
            "language": self.language,
            "source": self.source,
            "kind": self.kind,
            "needs_review": self.needs_review,
            "backend": self.backend,
            "voice": self.voice,
            "speed": self.speed,
            "pause_scale": self.pause_scale,
            "notes": self.notes,
        }
        self.config_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, root: Path) -> Project:
        root = Path(root)
        config = root / "project.json"
        if not config.exists():
            raise FileNotFoundError(f"{root} n'est pas un projet voxlibris ({config} absent)")
        return cls(root=root, **json.loads(config.read_text(encoding="utf-8")))

    @classmethod
    def create(cls, root: Path, document: Document) -> Project:
        """Crée le projet et y dépose l'extraction brute."""
        project = cls(
            root=Path(root),
            title=document.title,
            author=document.author,
            year=document.year,
            language=document.language,
            source=str(document.source) if document.source else None,
            kind=str(document.notes.get("kind", "")),
            needs_review=document.needs_review,
            notes={k: v for k, v in document.notes.items() if k != "dropped_lines"},
        )
        document.write(project.raw_dir)
        # Un texte qui n'a rien à relire peut servir directement de source de vérité.
        if not document.needs_review:
            document.write(project.clean_dir)
        project.save()
        return project

    # --- Contenu ----------------------------------------------------------------------
    def chapter_texts(self) -> dict[str, str]:
        """Le texte de chaque chapitre, tel que l'éditeur de relecture l'affiche.

        Passer par `Chapter` plutôt que de découper l'en-tête à la main garantit que les
        positions relevées ici désignent bien les mêmes caractères que ceux de la zone de
        saisie — sans quoi une suggestion s'appliquerait quelques caractères trop loin.
        """
        from .document import Chapter

        return {
            path.name: Chapter.from_markdown(path.read_text(encoding="utf-8")).text
            for path in sorted(self.text_dir.glob("ch*.md"))
        }

    # --- Édition ----------------------------------------------------------------------
    def delete_chapter(self, number: int) -> dict[str, int]:
        """Supprime un chapitre et renumérote les suivants.

        Les tables des matières se prennent souvent les pieds dans le tapis : une page de
        copyright arrive en tête et devient « chapitre 1 », si bien que tout le livre est
        décalé d'un cran. La voix annonce alors « Chapitre deux » avant un texte qui se
        présente comme le premier. Aucune heuristique ne rattrapera tous les cas — ce
        livre-ci numérote même deux chapitres « 7 » — et il faut donc pouvoir trancher
        à la main.

        Renuméroter invalide ce qui a été produit ensuite : les segments, et les pistes
        des chapitres à partir de celui qu'on retire, dont l'annonce est gravée dans le
        son. Les garder reviendrait à livrer un livre dont la voix se trompe de chapitre.
        """
        if not (self.raw_dir / f"ch{number:02d}.md").exists():
            raise FileNotFoundError(f"Le chapitre {number} n'existe pas.")

        for directory in (self.raw_dir, self.clean_dir):
            if not directory.exists():
                continue
            (directory / f"ch{number:02d}.md").unlink(missing_ok=True)
            for path in sorted(directory.glob("ch*.md")):
                current = int(path.stem.removeprefix("ch"))
                if current <= number:
                    continue
                # Seule la ligne de numéro change : le texte relu n'est pas réécrit.
                text = re.sub(
                    r"^chapter:\s*\d+",
                    f"chapter: {current - 1}",
                    path.read_text(encoding="utf-8"),
                    count=1,
                    flags=re.M,
                )
                path.write_text(text, encoding="utf-8")
                path.rename(directory / f"ch{current - 1:02d}.md")

        return self.invalidate_from(number)

    def merge_into_previous(self, number: int) -> dict[str, int]:
        """Recolle un chapitre au précédent, puis le supprime.

        Une illustration pleine page suffit à couper un chapitre en deux dans un EPUB :
        le fragment qui suit n'a ni titre ni début de phrase, et se ferait annoncer comme
        un chapitre à part entière, au milieu d'un mot. Le rattacher est la seule issue —
        aucune découpe automatique ne devinera qu'il fallait recoller là et pas ailleurs.
        """
        from .document import Chapter

        if number <= 1:
            raise ValueError("Le premier chapitre n'a pas de précédent.")

        for directory in (self.raw_dir, self.clean_dir):
            source = directory / f"ch{number:02d}.md"
            target = directory / f"ch{number - 1:02d}.md"
            if not (source.exists() and target.exists()):
                continue
            head = Chapter.from_markdown(target.read_text(encoding="utf-8"))
            tail = Chapter.from_markdown(source.read_text(encoding="utf-8"))
            head.paragraphs += tail.paragraphs
            if head.source_pages and tail.source_pages:
                head.source_pages = (head.source_pages[0], tail.source_pages[1])
            target.write_text(head.to_markdown(), encoding="utf-8")

        return self.delete_chapter(number)

    def invalidate_from(self, number: int) -> dict[str, int]:
        """Écarte les produits dérivés que la renumérotation a rendus faux."""
        counts = {"segments": 0, "pistes": 0, "assemblages": 0}
        for path in self.segments_dir.glob("ch*.jsonl"):
            path.unlink()
            counts["segments"] += 1
        for path in self.wav_dir.glob("ch*"):
            if int(path.name[2:4]) >= number:
                path.unlink()
                counts["pistes"] += 1
        for pattern in ("*.m4b", "mp3/*.mp3"):
            for path in self.out_dir.glob(pattern):
                path.unlink()
                counts["assemblages"] += 1
        return counts

    # --- État -----------------------------------------------------------------------
    def chapter_states(self) -> list[dict[str, object]]:
        """Pour chaque chapitre, son numéro, son titre et son état de relecture."""
        states = []
        for path in sorted(self.raw_dir.glob("ch*.md")):
            number = int(path.stem.removeprefix("ch"))
            reviewed = (self.clean_dir / path.name).exists()
            head = path.read_text(encoding="utf-8").split("---", 2)
            title = ""
            for line in head[1].splitlines() if len(head) > 2 else []:
                if line.startswith("title:"):
                    title = line.partition(":")[2].strip().strip('"')
            body = (self.clean_dir if reviewed else self.raw_dir) / path.name
            words = len(body.read_text(encoding="utf-8").split("---", 2)[-1].split())
            states.append(
                {"number": number, "title": title, "reviewed": reviewed, "words": words}
            )
        return states

    @property
    def pending_review(self) -> int:
        """Nombre de chapitres non relus, sur un texte qui en réclame une.

        Un texte océrisé synthétisé sans relecture fait lire à voix haute les coquilles
        de l'OCR — et, si la détection de plage a laissé passer des annexes, la table des
        matières elle-même. C'est une déconvenue coûteuse : la synthèse dure des dizaines
        de minutes avant qu'on l'entende.
        """
        if not self.needs_review:
            return 0
        return sum(1 for state in self.chapter_states() if not state["reviewed"])

    def status(self) -> dict[str, object]:
        wavs = sorted(self.wav_dir.glob("ch*.wav"))
        return {
            "chapitres": len(list(self.raw_dir.glob("ch*.md"))),
            "relu": self.clean_dir.exists() and any(self.clean_dir.glob("ch*.md")),
            "segments": sum(
                1 for path in self.segments_dir.glob("ch*.jsonl") for _ in path.open()
            ),
            "chapitres synthétisés": len(wavs),
            "m4b": next(iter(self.out_dir.glob("*.m4b")), None),
        }
