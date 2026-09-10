"""Ce qu'un livre dit de lui-même, lu sans l'extraire.

Au dépôt d'un fichier, l'interface préremplit titre, auteur et langue avant que
l'utilisateur ne lance l'import : il corrige plutôt qu'il ne saisit. Cette lecture
doit rester instantanée, elle ne touche donc qu'aux en-têtes — le manifeste d'un EPUB,
le dictionnaire d'informations d'un PDF, les premières lignes d'un texte — jamais au
contenu.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from .plain import MARKDOWN_HEADING

NS = {
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
    "dc": "http://purl.org/dc/elements/1.1/",
}
# Les textes du projet Gutenberg s'ouvrent sur une fiche « Title: … / Author: … ».
FIELD = re.compile(r"^\s*(title|titre|author|auteur|language|langue)\s*:\s*(.+?)\s*$", re.I)
HEAD_LINES = 60
# Calibre range parfois l'auteur « Nom, Prénom » : on le remet dans l'ordre de lecture.
INVERTED = re.compile(r"^([^,]+),\s*([^,]+)$")


def peek(path: str | Path) -> dict[str, str]:
    """Titre, auteur, langue et année d'un livre, autant qu'il les déclare.

    Ne renvoie que ce qui est trouvé : une clé absente laisse le champ à l'utilisateur.
    Une lecture qui échoue — fichier tronqué, format inattendu — renvoie un dictionnaire
    vide plutôt qu'une erreur : préremplir est une commodité, pas une étape.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".epub":
            found = _epub(path)
        elif suffix == ".pdf":
            found = _pdf(path)
        else:
            found = _plain(path)
    except Exception:
        return {}
    if match := INVERTED.match(found.get("author", "")):
        found["author"] = f"{match.group(2).strip()} {match.group(1).strip()}"
    return {k: v for k, v in found.items() if v}


def _epub(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as zf:
        container = ElementTree.fromstring(zf.read("META-INF/container.xml"))
        rootfile = container.find(".//c:rootfile", NS)
        opf = rootfile.get("full-path") if rootfile is not None else "content.opf"
        package = ElementTree.fromstring(zf.read(posixpath.normpath(opf)))

    def first(tag: str) -> str:
        return (package.findtext(f".//dc:{tag}", default="", namespaces=NS) or "").strip()

    return {
        "title": first("title"),
        "author": first("creator"),
        "language": first("language")[:2].lower(),
        "year": first("date")[:4],
    }


def _pdf(path: Path) -> dict[str, str]:
    import pymupdf

    with pymupdf.open(path) as doc:
        meta = doc.metadata or {}
    # « D:20180116160000 » : la date PDF commence par « D: » puis l'année.
    date = (meta.get("creationDate") or "").removeprefix("D:")[:4]
    return {
        "title": (meta.get("title") or "").strip(),
        "author": (meta.get("author") or "").strip(),
        "year": date if date.isdigit() else "",
    }


def _plain(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        lines = [handle.readline() for _ in range(HEAD_LINES)]
    for line in lines:
        if match := FIELD.match(line):
            key = match.group(1).lower()
            key = {"titre": "title", "auteur": "author", "langue": "language"}.get(key, key)
            found.setdefault(key, match.group(2))
    if "title" not in found:
        for line in lines:
            if match := MARKDOWN_HEADING.match(line.strip()):
                found["title"] = match.group(2)
                break
    if "language" in found:
        # « French » ou « fr » : on ne garde que ce qui ressemble à un code.
        found["language"] = {
            "french": "fr",
            "english": "en",
            "spanish": "es",
            "german": "de",
            "italian": "it",
        }.get(found["language"].lower(), found["language"][:2].lower())
    return found
