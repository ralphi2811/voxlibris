"""Transforme les chapitres relus en segments prêts à être synthétisés.

Deux problèmes distincts sont traités ici :

1. **Normalisation.** Ce qui se lit à l'œil ne se prononce pas. Les guillemets et les
   parenthèses n'ont pas de son, les tirets cadratins de dialogue non plus, les chiffres
   doivent être écrits en toutes lettres, et un mot tout en capitales est souvent épelé
   lettre par lettre par les modèles TTS (« TERRE ! » deviendrait « T, E, R, R, E »).

2. **Découpe.** XTTS déraille au-delà de ~250 caractères : voix qui dérive, fins de
   phrase répétées, parfois pur charabia. On coupe donc sur des frontières de phrases,
   jamais au milieu d'une réplique, et on note la pause à insérer après chaque segment
   pour que le silence porte la ponctuation plutôt que le modèle.

Sortie : work/segments/chNN.jsonl, un objet JSON par segment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

from num2words import num2words

# Couper une phrase en plein milieu produit un fragment sans fin suivi d'un fragment
# commençant en minuscule, et XTTS verbalise alors parfois la ponctuation (« point »).
# La limite est donc portée à 250, ce que le modèle encaisse sans peine, pour que la
# grande majorité des phrases passent d'un seul tenant.
MAX_CHARS = 250

# Regrouper plusieurs phrases dans un même segment revient à confier au modèle le soin
# de marquer le point qui les sépare — et il l'expédie. La découpe suit donc les phrases,
# de sorte que chaque point devienne un vrai silence, que l'on maîtrise.
#
# Une phrase très courte reste toutefois collée à la suivante : synthétisée seule, elle
# a produit par le passé des énoncés fautifs (« Le bateau. », dix caractères, cinq
# secondes de babil). C'est le seuil ci-dessous qui l'en empêche.
#
# Il vise ces fragments-là, et eux seuls. Placé trop haut — soixante caractères — il
# recollait des phrases entières et rendait la découpe inutile ; le contrôle qualité,
# lui, écarte déjà tout ce qui descend sous douze caractères.
MIN_SEGMENT_CHARS = 30

# Durée des silences insérés à la concaténation, en millisecondes.
PAUSE_SEGMENT = 250  # entre deux fragments d'une même phrase trop longue
PAUSE_SENTENCE = 380  # entre deux phrases — c'est lui qui fait entendre le point
PAUSE_PARAGRAPH = 800  # entre deux paragraphes
PAUSE_TITLE = 1200  # après l'annonce du chapitre
PAUSE_ELLIPSIS = 450  # remplace des points de suspension en fin de segment

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[«\"A-ZÀ-ÝŒ—])")
# Le motif doit se terminer sur un chiffre : sinon il avale l'espace suivant et colle
# le nombre écrit en lettres au mot d'après (« mille six cent cinquanteet précédait »).
NUMBER = re.compile(r"\d+(?:[\s.,]\d+)*")
ALL_CAPS = re.compile(r"\b[A-ZÀ-ÝŒ]{2,}\b")


def read_chapter(path: Path) -> tuple[dict[str, str], list[str]]:
    """Sépare l'en-tête YAML du corps et renvoie les paragraphes."""
    text = path.read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    meta = {}
    for line in front.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    paragraphs = [p.strip() for p in body.strip().split("\n\n") if p.strip()]
    return meta, paragraphs


def spell_number(match: re.Match[str]) -> str:
    """Écrit un nombre en toutes lettres (« 1650 » → « mille six cent cinquante »)."""
    raw = match.group().strip()
    digits = re.sub(r"[\s.,]", "", raw)
    if not digits:
        return raw
    return num2words(int(digits), lang="fr")


def normalize(text: str) -> str:
    """Rend un paragraphe prononçable."""
    # Espaces typographiques fines et insécables : invisibles à l'œil, perturbantes
    # pour le tokeniseur.
    text = "".join(" " if unicodedata.category(c) == "Zs" else c for c in text)

    text = text.replace("’", "'")
    text = text.replace("«", "").replace("»", "").replace("“", "").replace("”", "")
    # Les parenthèses deviennent des virgules : supprimées, l'incise se fondrait dans la
    # phrase porteuse et la rendrait incompréhensible à l'oreille.
    text = text.replace("(", ", ").replace(")", ", ")

    # Tiret de dialogue en tête de paragraphe : la pause entre paragraphes le remplace.
    text = re.sub(r"^[—–-]\s*", "", text)
    # Tout tiret cadratin restant est une incise : une virgule produit la même
    # respiration, et elle, elle se prononce.
    text = re.sub(r"[—–]", ",", text)

    text = NUMBER.sub(spell_number, text)
    # Un mot tout en capitales est lu comme un sigle par la plupart des modèles.
    text = ALL_CAPS.sub(lambda m: m.group().capitalize(), text)

    text = re.sub(r"\s+([,;:!?…])", r"\1", text)  # espace fine française avant la ponctuation
    text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)  # doublons créés par les remplacements
    text = re.sub(r"[,;:]+\s*([.!?…])", r"\1", text)
    text = re.sub(r"([.!?…])\s*[,;:]+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    # Comme pour les segments, seule la ponctuation faible de tête est retirée : un
    # deux-points final annonce une réplique (« L'aîné dit : ») et doit être conservé,
    # sans quoi le paragraphe se termine sans ponctuation du tout.
    return text.lstrip(" ,;:").rstrip()


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(paragraph) if s.strip()]


def hard_split(sentence: str) -> list[str]:
    """Coupe une phrase trop longue, en préférant la ponctuation faible aux espaces."""
    if len(sentence) <= MAX_CHARS:
        return [sentence]
    for pattern in (r"(?<=[;:])\s+", r"(?<=,)\s+"):
        parts = re.split(pattern, sentence)
        if len(parts) > 1:
            out: list[str] = []
            for part in parts:
                out.extend(hard_split(part.strip()))
            return out
    # Dernier recours : découpe sur les espaces, au plus près de la limite.
    words, chunks, current = sentence.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > MAX_CHARS:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


def segment_paragraph(paragraph: str) -> list[tuple[str, bool]]:
    """Découpe un paragraphe en segments, chacun marqué « finit une phrase » ou non.

    Un segment par phrase, sauf quand la phrase est trop courte pour être synthétisée
    seule — elle rejoint alors la suivante — ou trop longue pour le modèle, auquel cas
    elle est fragmentée et seul le dernier fragment clôt la phrase. Le drapeau renvoyé
    dit lequel des deux silences insérer.
    """
    segments: list[tuple[str, bool]] = []
    current = ""
    for sentence in split_sentences(paragraph):
        pieces = hard_split(sentence)
        for index, piece in enumerate(pieces):
            last_piece = index == len(pieces) - 1
            if current and len(current) + 1 + len(piece) > MAX_CHARS:
                segments.append((current, False))
                current = piece
            else:
                current = f"{current} {piece}".strip()
            # On ne clôt la phrase que si le segment a de quoi tenir debout tout seul.
            if last_piece and len(current) >= MIN_SEGMENT_CHARS:
                segments.append((current, True))
                current = ""
    if current:
        segments.append((current, True))
    return segments


def defuse_ellipsis(segment: str) -> tuple[str, int]:
    """Convertit les points de suspension en silence plutôt qu'en texte.

    Les modèles TTS les rendent mal : selon les cas ils les ignorent, les prononcent, ou
    partent en vrille sur la suite. Or une suspension n'est rien d'autre qu'une pause —
    autant la produire avec du vrai silence, que l'on maîtrise. En milieu de phrase une
    virgule suffit ; en fin de segment on retire les points et on allonge la pause.
    """
    extra = 0
    segment = re.sub(r"^\s*\.{2,}\s*", "", segment)
    if re.search(r"\.{2,}\s*$", segment):
        segment = re.sub(r"\s*\.{2,}\s*$", "", segment)
        extra = PAUSE_ELLIPSIS
    segment = re.sub(r"\s*\.{2,}\s*", ", ", segment)
    segment = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", segment)
    # Seule la ponctuation faible de tête est retirée. Celle de fin est conservée : quand
    # une phrase trop longue a dû être coupée, c'est la virgule de coupure, et l'ôter
    # laisserait un fragment sans aucune ponctuation finale — précisément ce qui pousse
    # XTTS à prononcer « point ».
    return segment.lstrip(" ,;:").rstrip(), extra


def announce(chapter: int, title: str) -> str:
    if chapter == 0:
        return ""
    spoken = f"Chapitre {num2words(chapter, lang='fr')}"
    named = (title or "").strip().rstrip(".").strip()
    # Un titre qui se réduit à « Chapitre N » n'est pas un titre : il ne dit rien que
    # l'annonce ne dise déjà. Le numéro qu'il porte n'est d'ailleurs pas fiable — les
    # tables des matières se trompent, en sautent, en répètent — et le suivre donnerait
    # « Chapitre six. Chapitre sept. » On ne garde donc que les titres qui nomment.
    if not named or re.fullmatch(r"chapitres?\s*\d+", named, re.I):
        named = ""
    # Le titre vient de l'en-tête YAML et n'a donc pas traversé normalize() : sans cet
    # appel, l'apostrophe typographique de « Mort d'un personnage » passe telle quelle.
    return normalize(f"{spoken}. {named}." if named else f"{spoken}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="text/clean")
    parser.add_argument("--out", default="work/segments")
    parser.add_argument(
        "--no-announce",
        action="store_true",
        help="ne pas faire annoncer « Chapitre un. La demande en mariage. » en tête de piste",
    )
    args = parser.parse_args()
    counts = build_segments(Path(args.dir), Path(args.out), announce_chapters=not args.no_announce)
    for chapter, count in sorted(counts.items()):
        print(f"ch{chapter:02d}  {count:>4} segments")
    print(f"\n{sum(counts.values())} segments au total.")


def build_chapter_segments(
    chapter: int,
    title: str,
    paragraphs: list[str],
    announce_chapter: bool = True,
    pause_scale: float = 1.0,
) -> list[dict[str, object]]:
    """Transforme les paragraphes d'un chapitre en segments prêts à synthétiser.

    `pause_scale` étire ou resserre tous les silences d'un même facteur. C'est le réglage
    à toucher quand la ponctuation ne s'entend pas assez — il agit sans resynthétiser
    quoi que ce soit, là où changer la vitesse oblige à tout refaire.
    """
    scale = max(0.25, min(4.0, float(pause_scale)))

    def silence(base: int) -> int:
        return int(round(base * scale))

    records: list[dict[str, object]] = []
    if announce_chapter and (header := announce(chapter, title)):
        records.append({"idx": 0, "text": header, "pause_after_ms": silence(PAUSE_TITLE)})

    for paragraph in paragraphs:
        segments = segment_paragraph(normalize(paragraph))
        for index, (segment, ends_sentence) in enumerate(segments):
            last = index == len(segments) - 1
            text, extra_pause = defuse_ellipsis(segment)
            if not text:
                continue
            if last:
                base = PAUSE_PARAGRAPH
            elif ends_sentence:
                base = PAUSE_SENTENCE
            else:
                base = PAUSE_SEGMENT
            records.append(
                {
                    "idx": len(records),
                    "text": text,
                    "pause_after_ms": silence(base) + silence(extra_pause),
                }
            )

    for record in records:
        record["chapter"] = chapter
        record["title"] = title
    return records


def build_segments(
    text_dir: Path,
    out_dir: Path,
    announce_chapters: bool = True,
    pause_scale: float = 1.0,
) -> dict[int, int]:
    """Écrit un JSONL de segments par chapitre ; renvoie le compte par chapitre."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[int, int] = {}
    for path in sorted(text_dir.glob("ch*.md")):
        meta, paragraphs = read_chapter(path)
        chapter = int(meta["chapter"])
        records = build_chapter_segments(
            chapter, meta["title"], paragraphs, announce_chapters, pause_scale
        )
        target = out_dir / f"ch{chapter:02d}.jsonl"
        content = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
        # Un fichier identique n'est pas réécrit : sa date reste celle du dernier vrai
        # changement, et c'est elle qui dira à la synthèse si la piste est à refaire.
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            target.write_text(content, encoding="utf-8")
        else:
            # Rien à réécrire, mais la préparation a bien eu lieu : sa date sert à savoir
            # si le texte a été corrigé depuis.
            os.utime(target)
        counts[chapter] = len(records)
    return counts


if __name__ == "__main__":
    main()
