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
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .assemble import BookMetadata
from .document import Document

# À côté de chaque piste, un petit fichier dit quelle voix l'a lue — moteur, voix et
# débit. Sans lui, une synthèse interrompue après un changement de voix laissait les
# chapitres non atteints passer pour à jour à la relance : ils disaient bien le texte,
# mais de l'ancienne voix.
STAMP_SUFFIX = ".voice"


def stamp_path(track: Path) -> Path:
    return track.with_suffix(STAMP_SUFFIX)


def read_stamp(track: Path) -> str:
    path = stamp_path(track)
    try:
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""
    except OSError:
        return ""


def write_stamp(track: Path, signature: str) -> None:
    stamp_path(track).write_text(signature, encoding="utf-8")


def make_signature(
    backend: str | None,
    voice: str | None,
    speed: float,
    voices: Mapping[str, str] | None = None,
) -> str:
    """Moteur, voix et débit — ce qui fait qu'une piste sonne comme les autres.

    La distribution s'ajoute au narrateur, persona par persona : « +dialogue=Ana
    +Charles=Damien ». Une piste faite avant qu'on la choisisse n'a donc pas la même
    note, et sera refaite pour ce que ces voix disent.
    """
    others = "".join(
        f"+{role}={chosen}"
        for role, chosen in sorted((voices or {}).items(), key=lambda kv: kv[0].casefold())
        if chosen and role.casefold() != "narrateur"
    )
    return f"{backend}/{voice}{others}@{speed:.2f}"


def parse_signature(signature: str) -> tuple[str, str, dict[str, str], str]:
    """Relit une note : moteur, narrateur, distribution, vitesse."""
    backend, _, rest = signature.partition("/")
    voices, _, speed = rest.rpartition("@")
    narrator, *parts = voices.split("+")
    cast: dict[str, str] = {}
    for part in parts:
        # Les premières notes à deux voix n'écrivaient que la voix des dialogues.
        role, _, chosen = part.partition("=") if "=" in part else ("dialogue", "=", part)
        cast[role] = chosen
    return backend, narrator, cast, speed


def same_engine(a: str, b: str) -> bool:
    """Deux notes du même moteur au même débit : leurs prises sont échangeables, voix
    par voix. C'est ce qui rend un changement de voix partiel."""
    if not (a and b):
        return False
    first, last = parse_signature(a), parse_signature(b)
    return (first[0], first[3]) == (last[0], last[3])


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
    # La distribution : persona → voix, sur le même moteur. « dialogue » est la voix des
    # répliques repérées à leur typographie ; les autres noms sont ceux que le texte
    # porte en marqueurs (« @Charles »). Un persona sans voix est lu par le narrateur.
    voices: dict[str, str] = field(default_factory=dict)
    # Débit de parole, et étirement de tous les silences. Les moteurs sont réglés pour
    # la phrase de démonstration, pas pour une heure d'écoute : à l'oreille, la lecture
    # court et la ponctuation s'efface. Ces deux réglages sont indépendants — la vitesse
    # oblige à resynthétiser, les pauses seulement à repréparer les segments.
    speed: float = 1.0
    pause_scale: float = 1.0
    # Annoncer « Chapitre trois — titre » en tête de chaque piste — le titre seul quand
    # le livre n'a qu'un chapitre. On le coupe pour un recueil dont les titres se suffisent.
    announce_chapters: bool = True
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
    def cover(self) -> Path | None:
        """Couverture déposée par l'utilisateur, si elle existe."""
        return next(iter(self.root.glob("cover.*")), None)

    @property
    def cover_source(self) -> Path | None:
        """D'où tirer la couverture : le fichier déposé, sinon le livre d'origine."""
        if self.cover:
            return self.cover
        return Path(self.source) if self.source else None

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
            "voices": self.voices,
            "speed": self.speed,
            "pause_scale": self.pause_scale,
            "announce_chapters": self.announce_chapters,
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
        data = json.loads(config.read_text(encoding="utf-8"))
        # Un projet réglé quand la seule seconde voix était celle des dialogues.
        if dialogue := data.pop("dialogue_voice", None):
            data.setdefault("voices", {})["dialogue"] = dialogue
        return cls(root=root, **data)

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
    def set_chapter_title(self, number: int, title: str) -> None:
        """Renomme un chapitre, dans le brut et dans le relu — le titre est annoncé à voix haute."""
        from .document import Chapter

        title = title.strip()
        if not title:
            raise ValueError("Un chapitre a besoin d'un titre.")
        found = False
        for directory in (self.raw_dir, self.clean_dir):
            path = directory / f"ch{number:02d}.md"
            if not path.exists():
                continue
            found = True
            chapter = Chapter.from_markdown(path.read_text(encoding="utf-8"))
            chapter.title = title
            path.write_text(chapter.to_markdown(), encoding="utf-8")
        if not found:
            raise FileNotFoundError(f"Le chapitre {number} n'existe pas.")

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
            states.append({"number": number, "title": title, "reviewed": reviewed, "words": words})
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

    @property
    def signature(self) -> str:
        """Moteur, voix et débit : ce qui fait qu'une piste sonne comme les autres."""
        return make_signature(self.backend, self.voice, self.speed, self.voices)

    def roles(self) -> Counter[str]:
        """Combien de segments chaque rôle a dans les segments préparés."""
        counts: Counter[str] = Counter()
        for path in self.segments_dir.glob("ch*.jsonl"):
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    counts[str(json.loads(line).get("role", "narrateur"))] += 1
        return counts

    def personas(self) -> list[str]:
        """Les personas du livre, hors narrateur et dialogues : ceux que le texte nomme,
        et ceux qu'une voix attend. Un même nom écrit à deux casses ne compte qu'une fois."""
        seen: dict[str, str] = {}
        for name in [*self.voices, *self.roles()]:
            if name.casefold() not in ("narrateur", "dialogue"):
                seen.setdefault(name.casefold(), name)
        return sorted(seen.values(), key=str.casefold)

    def track_signature(self, number: int) -> str:
        """La signature notée sur une piste ; vide pour une piste d'avant cette note."""
        return read_stamp(self.wav_dir / f"ch{number:02d}.wav")

    def timing(self, number: int) -> list[dict]:
        """Le manifeste d'une piste : chaque segment et sa plage dans le fichier."""
        path = self.wav_dir / f"ch{number:02d}.timing.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return []

    def text_path(self, number: int) -> Path:
        """Le texte qui fait foi pour un chapitre : relu s'il existe, brut sinon."""
        name = f"ch{number:02d}.md"
        return (self.clean_dir / name) if (self.clean_dir / name).exists() else self.raw_dir / name

    def segment_texts(self, number: int) -> list[tuple[int, str]]:
        path = self.segments_dir / f"ch{number:02d}.jsonl"
        if not path.exists():
            return []
        records = (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
        return [(int(r.get("idx", 0)), str(r.get("text", ""))) for r in records]

    def track_matches(self, number: int) -> bool:
        """Vrai si la piste dit exactement le texte des segments préparés.

        C'est le contenu qui décide, pas les dates : une préparation qui n'a rien changé
        ne périme aucune piste, et une piste ne peut pas passer pour à jour si un
        segment a changé sous elle.
        """
        timing = self.timing(number)
        if not timing or not (self.wav_dir / f"ch{number:02d}.wav").exists():
            return False
        # Une piste d'une autre voix n'est pas à jour, quand bien même elle dit le texte.
        stamped = self.track_signature(number)
        if stamped and stamped != self.signature:
            return False
        said = [(int(e.get("idx", 0)), str(e.get("text", ""))) for e in timing]
        return said == self.segment_texts(number)

    def text_changed(self, number: int) -> bool:
        """Vrai si le texte du chapitre a été enregistré après sa dernière préparation."""
        segments = self.segments_dir / f"ch{number:02d}.jsonl"
        text = self.text_path(number)
        if not segments.exists() or not text.exists():
            return False
        return text.stat().st_mtime > segments.stat().st_mtime

    def stale_chapters(self) -> list[int]:
        """Les chapitres corrigés depuis leur préparation."""
        return [
            int(state["number"])
            for state in self.chapter_states()
            if self.text_changed(int(state["number"]))
        ]

    def tracks(self) -> list[dict[str, object]]:
        """Chaque chapitre vu du côté de la synthèse : segments, piste, durée, alertes."""
        rows = []
        for state in self.chapter_states():
            number = int(state["number"])
            segments = self.segments_dir / f"ch{number:02d}.jsonl"
            track = self.wav_dir / f"ch{number:02d}.wav"
            timing = self.timing(number)
            count = 0
            if segments.exists():
                count = sum(1 for line in segments.open(encoding="utf-8") if line.strip())
            rows.append(
                {
                    **state,
                    "segments": count,
                    "synthesized": track.exists(),
                    "current": self.track_matches(number),
                    "text_changed": self.text_changed(number),
                    "seconds": timing[-1]["end"] if timing else 0.0,
                    "flagged": sum(1 for entry in timing if not entry.get("clean", True)),
                }
            )
        return rows

    def approve_segment(self, number: int, idx: int, approved: bool = True) -> bool:
        """Déclare un segment signalé bon à l'oreille — ou le remet en question.

        Le contrôle qualité juge sur des durées ; l'oreille tranche. Un segment validé sort
        de la liste à écouter et sera repris tel quel à la prochaine synthèse, au lieu
        d'être rejoué. Sa cause d'origine est conservée, pour savoir d'où il vient.
        """
        path = self.wav_dir / f"ch{number:02d}.timing.json"
        timing = self.timing(number)
        entry = next((e for e in timing if e.get("idx") == idx), None)
        if entry is None:
            return False
        if approved:
            entry["clean"], entry["approved"] = True, True
        elif entry.get("approved"):
            entry["clean"], entry["approved"] = False, False
        path.write_text(json.dumps(timing, ensure_ascii=False, indent=1), encoding="utf-8")
        return True

    def approved_segments(self) -> list[dict[str, object]]:
        """Les segments validés à l'oreille, pour pouvoir revenir dessus."""
        return [
            {"chapter": int(state["number"]), **entry}
            for state in self.chapter_states()
            for entry in self.timing(int(state["number"]))
            if entry.get("approved")
        ]

    def flagged_segments(self, context: int = 2) -> list[dict[str, object]]:
        """Les segments que le contrôle qualité n'a pas su rendre propres, toutes pistes.

        Chacun vient avec `before` et `after`, le texte des segments voisins : une phrase
        isolée ne dit pas toujours ce qui cloche, le passage autour si.
        """
        found = []
        for state in self.chapter_states():
            number = int(state["number"])
            timing = self.timing(number)
            for position, entry in enumerate(timing):
                if entry.get("clean", True):
                    continue
                before = timing[max(0, position - context) : position]
                after = timing[position + 1 : position + 1 + context]
                found.append(
                    {
                        "chapter": number,
                        **entry,
                        "before": " ".join(str(e.get("text", "")) for e in before),
                        "after": " ".join(str(e.get("text", "")) for e in after),
                    }
                )
        return found

    def audio_seconds(self) -> float:
        return sum(float(row["seconds"]) for row in self.tracks())

    def status(self) -> dict[str, object]:
        wavs = sorted(self.wav_dir.glob("ch*.wav"))
        return {
            "chapitres": len(list(self.raw_dir.glob("ch*.md"))),
            "relu": self.clean_dir.exists() and any(self.clean_dir.glob("ch*.md")),
            "segments": sum(1 for path in self.segments_dir.glob("ch*.jsonl") for _ in path.open()),
            "chapitres synthétisés": len(wavs),
            "m4b": next(iter(self.out_dir.glob("*.m4b")), None),
        }
