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
    "opf": "http://www.idpf.org/2007/opf",
}
# Recherche de couverture en ligne : Open Library, libre et sans clé. Seuls le titre
# et l'auteur partent sur le réseau, et seulement quand l'utilisateur le demande.
OPEN_LIBRARY = "https://openlibrary.org/search.json?{query}&limit=5&fields=cover_i"
OPEN_LIBRARY_COVER = "https://covers.openlibrary.org/b/id/{cover}-L.jpg?default=false"
USER_AGENT = "voxlibris (https://github.com/ralphi2811/voxlibris)"
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


def _package(zf: zipfile.ZipFile) -> tuple[ElementTree.Element, str]:
    """Le manifeste OPF d'un EPUB et le dossier où il vit."""
    container = ElementTree.fromstring(zf.read("META-INF/container.xml"))
    rootfile = container.find(".//c:rootfile", NS)
    opf = rootfile.get("full-path") if rootfile is not None else "content.opf"
    opf = posixpath.normpath(opf)
    return ElementTree.fromstring(zf.read(opf)), posixpath.dirname(opf)


def cover(path: str | Path) -> bytes | None:
    """L'image de couverture d'un EPUB, telle que le livre la désigne.

    Dans l'ordre : la balise `<meta name="cover">` du manifeste — par identifiant ou,
    façon Calibre, par nom de fichier —, l'entrée marquée `cover-image`, une image dont
    le nom dit « cover », et enfin, pour un EPUB issu d'un PDF, l'image de la première
    page.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            package, base = _package(zf)
            items = list(package.iterfind(".//opf:manifest/opf:item", NS))
            by_id = {item.get("id"): item for item in items}
            images = [i for i in items if (i.get("media-type") or "").startswith("image/")]
            found = None
            meta = package.find(".//opf:meta[@name='cover']", NS)
            # Un élément XML sans enfant compte pour faux : on teste toujours `is None`.
            if meta is not None and (wanted := meta.get("content")):
                found = by_id.get(wanted)
                if found is None:
                    found = next(
                        (i for i in images if (i.get("href") or "").endswith(wanted)), None
                    )
            if found is None:
                found = next(
                    (i for i in images if "cover-image" in (i.get("properties") or "")), None
                )
            if found is None:
                found = next(
                    (i for i in images if "cover" in (i.get("id", "") + i.get("href", "")).lower()),
                    None,
                )
            href = found.get("href") if found is not None else None
            if href is None:
                href = _first_page_image(zf, package, base, by_id)
            if not href:
                return None
            return zf.read(posixpath.normpath(posixpath.join(base, href)))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, ElementTree.ParseError):
        return None


def _first_page_image(zf, package, base: str, by_id: dict) -> str | None:
    first = package.find(".//opf:spine/opf:itemref", NS)
    item = by_id.get(first.get("idref")) if first is not None else None
    if item is None or not item.get("href"):
        return None
    page = posixpath.normpath(posixpath.join(base, item.get("href")))
    html = zf.read(page).decode("utf-8", errors="replace")
    match = re.search(r'<img\b[^>]*src="([^"]+)"', html, re.I)
    if not match:
        return None
    # Le chemin de l'image est relatif à la page, pas au manifeste.
    relative = posixpath.normpath(
        posixpath.join(posixpath.dirname(item.get("href")), match.group(1))
    )
    return relative


def search_cover(title: str, author: str = "", fetch=None) -> bytes | None:
    """Cherche une couverture sur Open Library ; None si rien de convaincant."""
    import json
    import urllib.parse
    import urllib.request

    if not title.strip():
        return None
    if fetch is None:

        def fetch(url: str) -> bytes:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.read()

    query = urllib.parse.urlencode({"title": title, **({"author": author} if author else {})})
    try:
        results = json.loads(fetch(OPEN_LIBRARY.format(query=query)))
        ids = [doc.get("cover_i") for doc in results.get("docs", []) if doc.get("cover_i")]
        for cover_id in ids[:3]:
            try:
                data = fetch(OPEN_LIBRARY_COVER.format(cover=cover_id))
            except OSError:
                continue
            # Une vraie image, pas le pixel de remplacement des couvertures absentes.
            if len(data) > 2000:
                return data
    except (OSError, ValueError):
        return None
    return None


def _epub(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as zf:
        package, _ = _package(zf)

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
