"""Détection du format d'un livre et aiguillage vers le lecteur adéquat.

La distinction délicate n'est pas entre EPUB et PDF — l'extension suffit — mais entre
les trois sortes de PDF, qui demandent des traitements sans rapport :

- **natif** : le texte vient d'un traitement de texte, il est exact, rien à relire ;
- **scanné avec couche OCR** : une reconnaissance de caractères a déjà été faite et
  posée sur l'image ; le texte existe mais comporte des fautes qu'il faut relire ;
- **scanné sans texte** : il n'y a que des images, l'OCR reste à faire.

Le signe distinctif entre les deux premiers est la police : les moteurs d'OCR posent
leur texte dans une police sans glyphes, nommée par convention `GlyphLessFont`, et dont
le rôle est justement d'être invisible au-dessus de l'image.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from ..document import Document, strip_gutenberg


class Kind(StrEnum):
    EPUB = "epub"
    PDF_TEXT = "pdf-texte"
    PDF_OCR = "pdf-scan-ocr"
    PDF_IMAGE = "pdf-scan-image"
    PLAIN = "texte"


PLAIN_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
# En deçà de ce nombre de caractères par page, un PDF n'a pas de couche de texte
# exploitable : ce qu'on y trouve est du bruit, pas un texte.
MIN_CHARS_PER_PAGE = 40
# Signature que voxlibris pose sur un PDF auquel il a ajouté lui-même la couche de texte
# (voir `ocr.py`) : le texte y est invisible, mais dans une police ordinaire.
OCR_PRODUCER = "voxlibris OCR"


class NeedsOCR(NotImplementedError):
    """Le PDF n'est qu'une suite d'images : il faut y lire les mots avant tout."""


def detect_kind(path: Path) -> Kind:
    """Détermine la nature d'un fichier, en l'ouvrant si son extension ne suffit pas."""
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return Kind.EPUB
    if suffix in PLAIN_SUFFIXES:
        return Kind.PLAIN
    if suffix != ".pdf":
        raise ValueError(f"Format non pris en charge : {suffix or path.name!r}")

    import pymupdf

    doc = pymupdf.open(path)
    # Un PDF que voxlibris a déjà lu est un scan océrisé, quoi qu'il en ait tiré : sans
    # cette signature, un scan presque vide repasserait pour une suite d'images à lire.
    if str((doc.metadata or {}).get("producer") or "").startswith(OCR_PRODUCER):
        return Kind.PDF_OCR
    sample = [doc[i] for i in range(0, len(doc), max(1, len(doc) // 12))][:12]
    chars = sum(len(page.get_text().strip()) for page in sample)
    if not sample or chars / len(sample) < MIN_CHARS_PER_PAGE:
        return Kind.PDF_IMAGE

    # `page.get_fonts()` révèle la police porteuse du texte. Une police sans glyphes
    # trahit une couche d'OCR posée sur une image.
    glyphless = any(
        "glyphless" in (font[3] or "").lower()
        for page in sample
        for font in page.get_fonts(full=True)
    )
    return Kind.PDF_OCR if glyphless else Kind.PDF_TEXT


def ingest(path: Path, **kwargs) -> Document:
    """Extrait un document, quel que soit son format d'origine."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    kind = detect_kind(path)
    if kind is Kind.EPUB:
        from . import epub

        document = epub.ingest(path, **kwargs)
    elif kind is Kind.PLAIN:
        from . import plain

        document = plain.ingest(path, **kwargs)
    elif kind is Kind.PDF_TEXT:
        from . import pdf_text

        document = pdf_text.ingest(path, **kwargs)
    elif kind is Kind.PDF_OCR:
        from . import pdf_scan

        document = pdf_scan.ingest(path, **kwargs)
    else:
        raise NeedsOCR(
            "Ce PDF ne contient aucune couche de texte : il faut d'abord y lire les mots. "
            "L'atelier le fait avec le service RapidOCR (profil « rapidocr » du Compose) ; "
            "en ligne de commande, `voxlibris ocr`."
        )

    document.notes["kind"] = kind.value
    if dropped := strip_gutenberg(document):
        document.notes["enveloppe Gutenberg retirée"] = f"{dropped} paragraphes"
    return document
