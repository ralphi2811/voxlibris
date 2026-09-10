"""Pages d'origine d'un livre, montrées en regard du texte pendant la relecture.

Pour un PDF, la page est rendue en image. Pour un EPUB issu de pdf2htmlEX, chaque page
est déjà un document HTML positionné au pixel, avec ses polices et l'image de fond du
PDF : plutôt que de la rastériser, on la sert telle quelle dans un cadre. On ne garde que
ce qu'il faut pour l'afficher — ses scripts et ses en-têtes de rafraîchissement sont
écartés, ses ressources redirigées vers l'atelier — et le cadre interdit toute exécution
de code, en-tête CSP à l'appui. La page est servie à sa taille réelle ; c'est la page de
relecture qui réduit le cadre à la largeur disponible.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree

from .ingest.epub_fixed import LAYOUT

# Seules les ressources d'affichage sortent du livre : ni pages, ni scripts, ni manifeste.
ASSET_TYPES = {
    ".otf": "font/otf",
    ".ttf": "font/ttf",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".css": "text/css",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}

SCRIPT = re.compile(r"<script\b[^>]*>.*?</script>|<script\b[^>]*/>", re.S | re.I)
META = re.compile(r"<meta\b[^>]*http-equiv[^>]*>", re.I)
HEAD = re.compile(r"<head\b[^>]*>", re.I)
VIEWPORT = re.compile(
    r'<meta\b[^>]*name="viewport"[^>]*content="([^"]*)"|'
    r'<meta\b[^>]*content="([^"]*)"[^>]*name="viewport"',
    re.I,
)
DIMENSION = re.compile(r"(width|height)\s*=\s*(\d+)")

# Rien ne doit défiler dans le cadre : la page fait exactement sa taille.
STYLE = "<style>html,body{margin:0;padding:0;overflow:hidden;background:#fff}</style>"

# Aucun script, et rien d'extérieur au livre : le cadre ne peut ni exécuter de code, ni
# appeler dehors, quoi que contienne l'EPUB.
CSP = "default-src 'none'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'"

NS = {
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
}


def kind(project) -> str | None:
    """« pdf », « epub » pour un EPUB paginé, sinon None : rien à montrer en regard."""
    if not project.source:
        return None
    suffix = Path(project.source).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".epub" and project.notes.get("layout") == LAYOUT:
        return "epub"
    return None


def size(html: str) -> tuple[int, int] | None:
    """Largeur et hauteur d'une page, d'après sa balise viewport ou son corps."""
    match = VIEWPORT.search(html)
    found = dict(DIMENSION.findall(match.group(1) or match.group(2) or "")) if match else {}
    if "width" in found and "height" in found:
        return int(found["width"]), int(found["height"])
    match = re.search(r"<body\b[^>]*width\s*:\s*(\d+)px[^>]*height\s*:\s*(\d+)px", html, re.I)
    return (int(match.group(1)), int(match.group(2))) if match else None


class Pages:
    """Les pages d'un EPUB paginé, dans l'ordre de lecture."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.base, self.names = _spine(self.path)

    def __len__(self) -> int:
        return len(self.names)

    def raw(self, index: int) -> str:
        with zipfile.ZipFile(self.path) as zf:
            return zf.read(self.names[index]).decode("utf-8", errors="replace")

    def page(self, index: int, assets_url: str) -> str:
        """La page prête à afficher : ressources servies par `assets_url`, scripts ôtés."""
        html = SCRIPT.sub("", self.raw(index))
        html = META.sub("", html)
        base = assets_url.rstrip("/") + "/" + (self.base + "/" if self.base else "")
        inject = f'<base href="{base}">' + STYLE
        match = HEAD.search(html)
        if match:
            return html[: match.end()] + inject + html[match.end() :]
        return inject + html

    def asset(self, relative: str) -> tuple[bytes, str] | None:
        """Une ressource du livre, ou None si elle n'existe pas ou n'a rien à faire ici."""
        name = posixpath.normpath(relative.lstrip("/"))
        media = ASSET_TYPES.get(posixpath.splitext(name)[1].lower())
        if media is None or name.startswith(".."):
            return None
        with zipfile.ZipFile(self.path) as zf:
            if name not in zf.namelist():
                return None
            return zf.read(name), media


@lru_cache(maxsize=16)
def _spine_cached(path: str, stamp: float) -> tuple[str, tuple[str, ...]]:
    with zipfile.ZipFile(path) as zf:
        container = ElementTree.fromstring(zf.read("META-INF/container.xml"))
        rootfile = container.find(".//c:rootfile", NS)
        opf_path = rootfile.get("full-path") if rootfile is not None else "content.opf"
        base = posixpath.dirname(opf_path)
        package = ElementTree.fromstring(zf.read(opf_path))
        hrefs = {
            item.get("id"): item.get("href")
            for item in package.iterfind(".//opf:manifest/opf:item", NS)
            if item.get("media-type") == "application/xhtml+xml"
        }
        names = []
        for ref in package.iterfind(".//opf:spine/opf:itemref", NS):
            href = hrefs.get(ref.get("idref"))
            if href:
                names.append(posixpath.normpath(posixpath.join(base, href)))
    return base, tuple(names)


def _spine(path: Path) -> tuple[str, list[str]]:
    """Ordre des pages, retenu tant que le fichier ne change pas."""
    stat = path.stat()
    base, names = _spine_cached(str(path), stat.st_mtime + stat.st_size)
    return base, list(names)
