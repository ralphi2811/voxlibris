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

    # --- État -----------------------------------------------------------------------
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
