"""Interface web de voxlibris.

Rendu côté serveur, HTMX pour les mises à jour partielles, aucune étape de compilation.
L'application ne fait aucun travail lourd elle-même : elle dépose des tâches dans la file
et affiche leur avancement. C'est ce qui permet de n'installer CUDA que dans l'ouvrier.

    uvicorn voxlibris.web.app:app --reload
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..document import Chapter
from ..ingest import ingest as ingest_book
from ..project import Project
from ..tts.backends import BACKENDS
from .jobs import default_queue, workspace

BASE = Path(__file__).parent
templates = Jinja2Templates(directory=BASE / "templates")
app = FastAPI(title="voxlibris")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

queue = default_queue()


# --- Utilitaires --------------------------------------------------------------------
def slug(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return cleaned or "livre"


def project_dir(name: str) -> Path:
    """Résout un nom de projet, en refusant toute évasion hors de l'espace de travail."""
    root = workspace()
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_dir():
        raise HTTPException(404, "Projet introuvable")
    return candidate


def load_project(name: str) -> Project:
    try:
        return Project.load(project_dir(name))
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error


def all_projects() -> list[tuple[str, Project]]:
    root = workspace()
    root.mkdir(parents=True, exist_ok=True)
    found = []
    for path in sorted(root.iterdir()):
        if (path / "project.json").exists():
            found.append((path.name, Project.load(path)))
    return found


def render(request: Request, template: str, **context) -> HTMLResponse:
    return templates.TemplateResponse(request, template, context)


_voxtral_cache: dict[str, list[str]] = {}


def catalogues(language: str = "") -> dict[str, list[str]]:
    """Voix proposables sans charger le moindre modèle, par moteur.

    XTTS n'en fournit aucune : ses locuteurs ne se lisent qu'une fois les huit
    gigaoctets en mémoire, ce que l'interface n'a pas à faire — d'où le champ libre qui
    subsiste pour lui. Voxtral, lui, tient son catalogue derrière une simple requête ;
    on la mémorise, car elle serait sinon refaite à chaque affichage de page.

    Une absence de clé ou un service en panne ne laissent qu'une liste vide : on retombe
    alors sur la saisie libre, plutôt que d'empêcher l'affichage du projet.
    """
    known = {name: list(cls.catalogue) for name, cls in BACKENDS.items()}
    if language not in _voxtral_cache:
        try:
            from ..tts.voxtral import voice_names

            _voxtral_cache[language] = voice_names(language)
        except Exception:
            _voxtral_cache[language] = []
    known["voxtral"] = _voxtral_cache[language]
    return known


def voxtral_is_local() -> bool:
    """Vrai si Voxtral est servi chez soi : ni facture ni envoi du texte à annoncer."""
    from ..tts.voxtral import DEFAULT_BASE_URL, base_url

    return "api.mistral.ai" not in (base_url() or DEFAULT_BASE_URL)


# Voix XTTS proposées d'emblée au banc d'essai. Elles ne peuvent pas être listées sans
# charger le modèle, et ce sont de toute façon celles qui tiennent le français.
XTTS_SUGGESTIONS = ("Viktor Menelaos", "Damien Black", "Tammie Ema")

# Nombre de voix retenues par moteur : un banc d'essai de trente extraits ne se compare
# pas à l'oreille, et chacun se paie chez Voxtral.
PER_BACKEND = 4


def sample_candidates(language: str = "") -> list[tuple[str, str, str, bool]]:
    """Voix à proposer au banc d'essai : (moteur, voix, intitulé, cochée d'avance)."""
    rows: list[tuple[str, str, str, bool]] = [
        ("xtts", voice, f"XTTS · {voice}", index < 2)
        for index, voice in enumerate(XTTS_SUGGESTIONS)
    ]
    for backend, voices in catalogues(language).items():
        if backend == "xtts":
            continue
        for voice in voices[:PER_BACKEND]:
            rows.append((backend, voice, f"{backend.capitalize()} · {voice}", False))
    return rows


# --- Projets ------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return render(request, "index.html", projects=all_projects())


@app.post("/projects")
async def create_project(file: UploadFile, title: str = Form(""), author: str = Form("")):
    """Dépose un livre, l'extrait, et crée le projet."""
    root = workspace()
    incoming = root / "_uploads"
    incoming.mkdir(parents=True, exist_ok=True)
    source = incoming / (file.filename or "livre")
    with source.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    kwargs = {k: v for k, v in {"title": title, "author": author}.items() if v}
    try:
        document = ingest_book(source, **kwargs)
    except (ValueError, NotImplementedError) as error:
        raise HTTPException(400, str(error)) from error

    name = slug(document.title)
    directory = root / name
    suffix = 2
    while directory.exists():
        directory = root / f"{name}-{suffix}"
        suffix += 1

    Project.create(directory, document)
    return RedirectResponse(f"/projects/{directory.name}", status_code=303)


@app.get("/projects/{name}", response_class=HTMLResponse)
def show_project(request: Request, name: str):
    project = load_project(name)
    return render(
        request,
        "project.html",
        name=name,
        project=project,
        chapters=project.chapter_states(),
        status=project.status(),
        job=queue.active(name),
        jobs=queue.list(name, limit=8),
        backends=sorted(BACKENDS),
        catalogues=catalogues(project.language),
        voxtral_local=voxtral_is_local(),
        candidates=sample_candidates(project.language),
        suggestions=len(load_suggestions(project)),
        samples=sorted((project.out_dir / "samples").glob("*.wav")),
    )


@app.post("/projects/{name}/delete")
def delete_project(name: str):
    """Supprime un projet et tout ce qu'il contient.

    POST plutôt que DELETE : un formulaire HTML ne sait envoyer que GET et POST, et
    passer par du JavaScript pour un bouton relèverait de l'entêtement.

    La suppression emporte le texte relu et les heures de synthèse — un livre de
    quatre-vingt-dix minutes en demande vingt à produire. Une tâche en cours écrirait
    d'ailleurs dans un dossier disparu : on refuse tant qu'elle n'est pas finie.
    """
    directory = project_dir(name)
    if queue.active(name):
        raise HTTPException(409, "Une tâche est en cours sur ce projet : attendez sa fin.")
    shutil.rmtree(directory)
    return RedirectResponse("/", status_code=303)


# --- Relecture ----------------------------------------------------------------------
def load_suggestions(project: Project) -> list:
    """Propositions du modèle, si une passe a été lancée.

    Un rapport illisible ou écrit par une version antérieure ne doit pas empêcher de
    relire : dans le doute, la page s'affiche sans suggestions plutôt qu'en erreur.
    """
    from ..proofread import Report

    path = project.suggestions_file
    if not path.exists():
        return []
    try:
        return Report.from_json(path.read_text(encoding="utf-8")).suggestions
    except (ValueError, TypeError):
        return []


@app.get("/projects/{name}/review/{number}", response_class=HTMLResponse)
def review_chapter(request: Request, name: str, number: int):
    """Éditeur de relecture : le texte à corriger, la page d'origine en regard."""
    from ..review import inspect_texts

    project = load_project(name)
    path = project.text_dir / f"ch{number:02d}.md"
    if not path.exists():
        raise HTTPException(404, "Chapitre introuvable")

    chapter = Chapter.from_markdown(path.read_text(encoding="utf-8"))
    suspects = inspect_texts({path.name: chapter.text}, language=project.language)
    numbers = sorted(int(p.stem.removeprefix("ch")) for p in project.text_dir.glob("ch*.md"))
    return render(
        request,
        "review.html",
        name=name,
        project=project,
        chapter=chapter,
        suspects=suspects,
        suggestions=[s for s in load_suggestions(project) if s.chapter == path.name],
        numbers=numbers,
        previous=max([n for n in numbers if n < number], default=None),
        following=min([n for n in numbers if n > number], default=None),
    )


@app.post("/projects/{name}/review/{number}")
def save_chapter(name: str, number: int, text: str = Form(...), title: str = Form(...)):
    """Enregistre la version relue, sans jamais écraser l'extraction brute."""
    project = load_project(name)
    source = project.text_dir / f"ch{number:02d}.md"
    chapter = Chapter.from_markdown(source.read_text(encoding="utf-8"))
    chapter.title = title.strip() or chapter.title
    chapter.paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]

    project.clean_dir.mkdir(parents=True, exist_ok=True)
    (project.clean_dir / f"ch{number:02d}.md").write_text(chapter.to_markdown(), encoding="utf-8")
    return RedirectResponse(f"/projects/{name}/review/{number}", status_code=303)


@app.post("/projects/{name}/chapters/{number}/{action}")
def edit_chapter(name: str, number: int, action: str):
    """Retire un chapitre, ou le recolle au précédent. Les suivants sont renumérotés."""
    if action not in {"delete", "merge"}:
        raise HTTPException(404, "Action inconnue")
    project = load_project(name)
    if queue.active(name):
        raise HTTPException(409, "Une tâche est en cours sur ce projet : attendez sa fin.")
    try:
        if action == "merge":
            project.merge_into_previous(number)
        else:
            project.delete_chapter(number)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return RedirectResponse(f"/projects/{name}", status_code=303)


@app.get("/projects/{name}/page/{number}")
def chapter_page_image(name: str, number: int, page: int = 0):
    """Rend une page du PDF d'origine, pour la relecture côte à côte."""
    project = load_project(name)
    if not project.source or not Path(project.source).suffix.lower() == ".pdf":
        raise HTTPException(404, "Pas de source paginée pour ce projet")

    chapter = Chapter.from_markdown(
        (project.text_dir / f"ch{number:02d}.md").read_text(encoding="utf-8")
    )
    if not chapter.source_pages:
        raise HTTPException(404, "Chapitre sans pages d'origine")

    first, last = chapter.source_pages
    wanted = min(max(first + page, first), last)
    cache = project.root / "work" / "pages"
    cache.mkdir(parents=True, exist_ok=True)
    image = cache / f"p{wanted:04d}.png"
    if not image.exists():
        import pymupdf

        pymupdf.open(project.source)[wanted - 1].get_pixmap(dpi=150).save(image)
    return FileResponse(image, media_type="image/png")


# --- Tâches -------------------------------------------------------------------------
def _number(raw: object, low: float, high: float) -> float | None:
    """Lit un réglage numérique du formulaire, borné. Renvoie None s'il est absent.

    Une valeur illisible est traitée comme absente : le réglage précédent du projet est
    alors conservé, plutôt que de faire échouer une tâche de vingt minutes sur une
    virgule mal placée.
    """
    if raw in (None, ""):
        return None
    try:
        return max(low, min(high, float(str(raw).replace(",", "."))))
    except ValueError:
        return None


@app.post("/projects/{name}/jobs/{kind}")
async def enqueue_job(request: Request, name: str, kind: str):
    load_project(name)
    if queue.active(name):
        raise HTTPException(409, "Une tâche est déjà en cours pour ce projet")

    form = await request.form()
    params: dict = {}
    if kind == "synth":
        params = {
            "backend": str(form.get("backend", "xtts")),
            "voice": str(form.get("voice") or "") or None,
            "force": bool(form.get("force")),
            "speed": _number(form.get("speed"), 0.7, 1.3),
        }
    elif kind == "normalize":
        params = {"pause_scale": _number(form.get("pause_scale"), 0.5, 3.0)}
    elif kind == "sample":
        picks = [str(v) for v in form.getlist("voice")]
        params = {
            "voices": [
                {"backend": b, "voice": v} for b, _, v in (pick.partition("/") for pick in picks)
            ]
        }
    elif kind == "assemble":
        params = {"skip_mp3": bool(form.get("skip_mp3"))}

    queue.enqueue(name, kind, **params)
    return RedirectResponse(f"/projects/{name}", status_code=303)


@app.get("/projects/{name}/progress", response_class=HTMLResponse)
def job_progress(request: Request, name: str):
    """Fragment rafraîchi par HTMX : état de la tâche courante."""
    return render(
        request,
        "partials/progress.html",
        name=name,
        job=queue.active(name),
        jobs=queue.list(name, limit=8),
        status=load_project(name).status(),
    )


# --- Fichiers produits --------------------------------------------------------------
@app.get("/projects/{name}/files/{path:path}")
def download(name: str, path: str):
    project = load_project(name)
    target = (project.out_dir / path).resolve()
    if not target.is_relative_to(project.out_dir.resolve()) or not target.is_file():
        raise HTTPException(404, "Fichier introuvable")
    return FileResponse(target, filename=target.name)


@app.get("/api/voices/{backend}")
def list_voices(backend: str):
    """Voix d'un moteur. Sollicité par l'interface au changement de moteur."""
    from ..tts.backends import load

    try:
        return {"voices": load(backend).voices()}
    except Exception as error:
        # Un moteur non installé ne doit pas casser la page : on le dit, simplement.
        return {"voices": [], "error": str(error)}
