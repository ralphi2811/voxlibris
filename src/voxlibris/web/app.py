"""Interface web de voxlibris — l'Atelier.

Rendu côté serveur, quelques fragments rafraîchis par sondage, aucune étape de
compilation. L'application ne fait aucun travail lourd elle-même : elle dépose des tâches
dans la file et affiche leur avancement. C'est ce qui permet de n'installer CUDA que dans
l'atelier, le processus qui les exécute.

L'interface suit la chaîne du livre audio, une page par étape : chapitres, relecture,
préparation, voix, synthèse, assemblage. La barre latérale porte l'état de chaque étape,
pour qu'on sache d'un coup d'œil où en est un livre et ce qui reste à faire.

    uvicorn voxlibris.web.app:app --reload
"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import atelier, pages
from ..concierge import served_state
from ..config import EDITABLE, SECRETS, origin, setting, voices_dir, write_settings
from ..document import Chapter
from ..ingest import ingest as ingest_book
from ..project import Project
from ..tts.backends import BACKENDS, SPEED_RANGE
from .jobs import State, default_queue, workspace

BASE = Path(__file__).parent
templates = Jinja2Templates(directory=BASE / "templates")
app = FastAPI(title="voxlibris")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


def static_version() -> str:
    """Empreinte du script et de la feuille de style, à coller à leur adresse.

    Sans elle, un navigateur garde l'ancien script des jours durant après une mise à
    jour, sans rien demander au serveur — et l'interface se comporte comme avant.
    """
    digest = hashlib.sha1()
    for name in ("app.js", "style.css"):
        digest.update((BASE / "static" / name).read_bytes())
    return digest.hexdigest()[:10]


templates.env.globals["static_version"] = static_version()

queue = default_queue()

# Les moteurs, tels que l'interface les présente. La licence et le matériel sont ce qui
# décide en pratique : un usage commercial exclut XTTS, une machine sans carte exclut
# XTTS et Voxtral local.
ENGINES = {
    "xtts": {
        "label": "XTTS-v2",
        "blurb": "La meilleure prosodie en français. Le défaut.",
        "licence": "CPML · non commercial",
        "licence_tone": "warn",
        "where": "GPU · 4 Go",
    },
    "kokoro": {
        "label": "Kokoro-82M",
        "blurb": "Rapide et très stable, une voix française.",
        "licence": "Apache 2.0",
        "licence_tone": "ok",
        "where": "GPU ou CPU",
    },
    "piper": {
        "label": "Piper",
        "blurb": "Quasi instantané : pour régler pauses et vitesse avant la version finale.",
        "licence": "MIT",
        "licence_tone": "ok",
        "where": "CPU",
    },
    "voxtral": {
        "label": "Voxtral TTS",
        "blurb": "Neuf langues. Chez Mistral, ou chez vous via vLLM.",
        "licence": "CC BY-NC 4.0",
        "licence_tone": "warn",
        "where": "API ou GPU · 16 Go",
    },
    "zonos2": {
        "label": "ZONOS2",
        "blurb": "Clone une voix d'un simple extrait. Chez vous, quarante langues.",
        "licence": "Apache 2.0",
        "licence_tone": "ok",
        "where": "GPU · 24 Go",
    },
    "omnivoice": {
        "label": "OmniVoice",
        "blurb": "Clone une voix d'un extrait de dix secondes. Léger, six cents langues.",
        "licence": "Apache 2.0",
        "licence_tone": "ok",
        "where": "GPU · 3 Go",
    },
}

# Voix XTTS proposées d'emblée au banc d'essai : elles ne peuvent pas être listées sans
# charger le modèle, et ce sont de toute façon celles qui tiennent le français.
XTTS_SUGGESTIONS = ("Viktor Menelaos", "Damien Black", "Tammie Ema")

# Nombre de voix retenues par moteur au banc d'essai : trente extraits ne se comparent
# pas à l'oreille, et chacun se paie chez Voxtral.
PER_BACKEND = 4

# Page où l'on atterrit après une tâche, selon sa nature.
LANDING = {
    "normalize": "prepare",
    "proofread": "review/1",
    "sample": "voices",
    "synth": "synth",
    "resynth": "synth",
    "assemble": "assemble",
}


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


def minutes(seconds: float) -> str:
    """« 1 h 04 » ou « 51 min » : la durée telle qu'on la lit sur une jaquette."""
    total = int(round(seconds / 60))
    if total >= 60:
        return f"{total // 60} h {total % 60:02d}"
    return f"{total} min"


def clock(seconds: float) -> str:
    """« 12:34 » ou « 1:02:03 » : un instant dans une piste."""
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    mins, secs = divmod(rest, 60)
    return f"{hours}:{mins:02d}:{secs:02d}" if hours else f"{mins}:{secs:02d}"


templates.env.filters["minutes"] = minutes
templates.env.filters["clock"] = clock


# Les catalogues servis par une adresse — Voxtral, ZONOS2, OmniVoice — sont gardés en
# mémoire un court instant. Une page les demande plusieurs fois, et chercher un serveur
# arrêté coûte à chaque fois plusieurs secondes de résolution de nom, que le délai de
# connexion ne borne pas : sans mémoire, la page Voix mettait sept secondes à venir, le
# serveur ZONOS2 éteint. Trente secondes suffisent pour voir un serveur démarré ; un
# dépôt ou un retrait de voix vide la mémoire de lui-même.
CATALOGUE_TTL = 30.0
# Un serveur qui met plus de trois secondes à lister ses voix est tenu pour absent : la
# page ne l'attend pas. La synthèse, elle, garde son propre délai, bien plus long.
PROBE_TIMEOUT = 3.0
_catalogue_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}
# Les sondes partent ensemble, dans des fils : la page attend le plus lent, pas la somme,
# et pas au-delà du délai — une résolution de nom qui traîne finit seule, dans son fil.
_probes = ThreadPoolExecutor(max_workers=3, thread_name_prefix="catalogue")
REMOTE_ENGINES = ("voxtral", "zonos2", "omnivoice")


def _fetch(engine: str, language: str) -> list[str]:
    # Un serveur en veille ne se sonde pas : on le réveillera pour la tâche. Ses voix,
    # pour les cloneurs, sont les fichiers du dossier — on les connaît sans lui.
    if served_state(engine) == "stopped":
        return sample_labels() if engine in ("zonos2", "omnivoice") else []
    if engine == "voxtral":
        return voxtral_voices(language)
    return {"zonos2": zonos2_voices, "omnivoice": omnivoice_voices}[engine]()


def remote_catalogues(language: str = "") -> dict[str, list[str]]:
    """Les catalogues servis par une adresse, mémorisés le temps d'une visite."""
    now = time.monotonic()
    found: dict[str, list[str]] = {}
    pending: dict[str, Future] = {}
    for engine in REMOTE_ENGINES:
        hit = _catalogue_cache.get((engine, language))
        if hit and now - hit[0] < CATALOGUE_TTL:
            found[engine] = hit[1]
        else:
            pending[engine] = _probes.submit(_fetch, engine, language)
    deadline = time.monotonic() + PROBE_TIMEOUT
    for engine, future in pending.items():
        try:
            names = future.result(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:  # serveur absent, trop lent, ou clé manquante : liste vide
            names = []
        _catalogue_cache[(engine, language)] = (now, names)
        found[engine] = names
    return found


def remote_voices(engine: str, language: str = "") -> list[str]:
    """Le catalogue d'un seul moteur servi par une adresse."""
    return remote_catalogues(language).get(engine, [])


def forget_catalogues() -> None:
    """À appeler quand les voix ont changé sous nos yeux : la prochaine page redemande."""
    _catalogue_cache.clear()


def catalogues(language: str = "") -> dict[str, list[str]]:
    """Voix proposables sans charger le moindre modèle, par moteur.

    XTTS n'en fournit aucune : ses locuteurs ne se lisent qu'une fois les huit
    gigaoctets en mémoire, ce que l'interface n'a pas à faire — d'où les suggestions
    fixes. Voxtral, ZONOS2 et OmniVoice tiennent le leur derrière une requête, mémorisée
    un court instant (voir `remote_voices`).

    Une absence de clé ou un service en panne ne laissent qu'une liste vide : on retombe
    alors sur la saisie libre, plutôt que d'empêcher l'affichage du projet.
    """
    known = {name: list(cls.catalogue) for name, cls in BACKENDS.items()}
    known["xtts"] = list(XTTS_SUGGESTIONS)
    known.update(remote_catalogues(language))
    return known


def voxtral_voices(language: str = "") -> list[str]:
    """Les voix Voxtral pour une langue — rien sans clé ni serveur local."""
    from ..tts.voxtral import Client, voice_names

    return voice_names(language, Client(timeout=PROBE_TIMEOUT))


# Les voix anglaises livrées avec le serveur, posées dans le dossier au premier départ.
# Elles passent derrière celles qu'on a déposées : pour un livre français, ce sont les
# siennes qu'on veut voir au banc, pas trois voix d'exemple.
ZONOS2_SHIPPED = ("AmericanFemale", "AmericanMale", "BritishFemale")


def zonos2_voices() -> list[str]:
    """Les voix que voit le serveur ZONOS2 — rien, s'il n'est pas configuré ou absent."""
    from ..tts.zonos2 import Client, base_url, voice_names

    if not base_url():
        return []
    try:
        names = voice_names(Client(timeout=PROBE_TIMEOUT))
    except Exception:
        return []
    return sorted(names, key=lambda n: (n in ZONOS2_SHIPPED, n.lower()))


def omnivoice_voices() -> list[str]:
    """Les voix que voit le serveur OmniVoice — rien, s'il n'est pas configuré ou absent."""
    from ..tts.omnivoice import Client, base_url, voice_names

    if not base_url():
        return []
    try:
        names = voice_names(Client(timeout=PROBE_TIMEOUT))
    except Exception:
        return []
    return sorted(names, key=lambda n: (n in ZONOS2_SHIPPED, n.lower()))


def sample_labels() -> list[str]:
    """Les intitulés que les serveurs de clonage donneront aux extraits déposés."""
    names = [v["label"] for v in voice_samples()]
    return sorted(names, key=lambda n: (n in ZONOS2_SHIPPED, n.lower()))


def voice_samples() -> list[dict[str, str]]:
    """Les extraits déposés pour le clonage, tels que le serveur les nommera."""
    from ..tts.zonos2 import AUDIO_EXTENSIONS

    folder = voices_dir()
    if not folder.is_dir():
        return []
    return [
        {
            "file": path.name,
            "label": path.stem.replace("_", " ").replace("-", " ").strip() or path.name,
        }
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]


def voxtral_is_local() -> bool:
    """Vrai si Voxtral est servi chez soi : ni facture ni envoi du texte à annoncer."""
    from ..tts.voxtral import DEFAULT_BASE_URL, base_url

    return "api.mistral.ai" not in (base_url() or DEFAULT_BASE_URL)


def sample_candidates(
    language: str = "", project: Project | None = None, known: dict[str, list[str]] | None = None
) -> list[tuple[str, str, str, bool]]:
    """Voix à proposer au banc d'essai : (moteur, voix, intitulé, cochée d'avance).

    Deux voix XTTS sont cochées pour un projet qui n'a encore rien écouté ni choisi :
    c'est un point de départ. Dès qu'un extrait existe ou qu'un moteur est retenu, plus
    rien n'est coché d'office — surtout pas ce qui a déjà été produit.
    """
    fresh = project is None or (not project.backend and not samples_of(project))
    done = (
        set() if project is None else {(s["backend"], s["voice_raw"]) for s in samples_of(project)}
    )
    rows: list[tuple[str, str, str, bool]] = [
        ("xtts", voice, f"XTTS · {voice}", fresh and index < 2 and ("xtts", voice) not in done)
        for index, voice in enumerate(XTTS_SUGGESTIONS)
    ]
    for backend, voices in (known or catalogues(language)).items():
        if backend == "xtts":
            continue
        for voice in voices[:PER_BACKEND]:
            rows.append((backend, voice, f"{ENGINES[backend]['label']} · {voice}", False))
    return rows


def samples_of(project: Project) -> list[dict[str, object]]:
    """Les extraits du banc d'essai déjà produits, avec leur moteur et leur voix."""
    found = []
    for path in sorted((project.out_dir / "samples").glob("*.wav")):
        backend, _, voice = path.stem.partition("--")
        found.append(
            {
                "backend": backend,
                "label": ENGINES.get(backend, {}).get("label", backend),
                "voice": voice.replace("_", " "),
                "voice_raw": voice.replace("_", " "),
                "file": f"samples/{path.name}",
                "chosen": project.backend == backend
                and (project.voice or "").replace(" ", "_") == voice,
            }
        )
    return found


# --- Coquille : barre latérale et état des étapes ---------------------------------------
def steps_for(name: str, project: Project, active: str) -> list[dict[str, object]]:
    """L'état de chaque étape de la chaîne, tel que la barre latérale l'affiche."""
    status = project.status()
    tracks = project.tracks()
    chapters = int(status["chapitres"])
    pending = project.pending_review
    running = queue.active(name)

    def step(key: str, label: str, href: str, done: bool, note: str, tone: str = "") -> dict:
        state = "on" if key == active else ("done" if done else "todo")
        return {
            "key": key,
            "label": label,
            "href": href,
            "state": state,
            "note": note,
            "tone": tone,
        }

    if not project.needs_review:
        review_note, review_tone = "facultative", ""
    elif pending:
        review_note, review_tone = f"{pending} à relire", "warn"
    else:
        review_note, review_tone = "relu", ""

    current = sum(1 for t in tracks if t["current"])
    if running and running.kind in ("synth", "resynth"):
        synth_note, synth_tone = "en cours", "accent"
    elif not tracks:
        synth_note, synth_tone = "—", ""
    else:
        synth_note, synth_tone = f"{current}/{len(tracks)} pistes", ""

    return [
        step("chapters", "Chapitres", f"/projects/{name}", chapters > 0, f"{chapters}"),
        step(
            "review",
            "Relecture",
            f"/projects/{name}/review/1",
            bool(status["relu"]) and not pending,
            review_note,
            review_tone,
        ),
        step(
            "prepare",
            "Préparation",
            f"/projects/{name}/prepare",
            int(status["segments"]) > 0,
            f"{status['segments']} seg." if status["segments"] else "—",
        ),
        step(
            "voices",
            "Voix",
            f"/projects/{name}/voices",
            bool(project.voice),
            project.voice or "à choisir",
        ),
        step(
            "synth",
            "Synthèse",
            f"/projects/{name}/synth",
            bool(tracks) and current == len(tracks),
            synth_note,
            synth_tone,
        ),
        step(
            "assemble",
            "Assemblage",
            f"/projects/{name}/assemble",
            status["m4b"] is not None,
            "M4B prêt" if status["m4b"] else "—",
        ),
    ]


def shell(request: Request, template: str, name: str | None = None, active: str = "", **extra):
    """Rend une page dans la coquille : barre latérale, atelier, tâche en cours."""
    project = load_project(name) if name else None
    context = {
        "name": name,
        "project": project,
        "active": active,
        "steps": steps_for(name, project, active) if name and project else [],
        "atelier": atelier.status(workspace()),
        "job": queue.active(name) if name else None,
        **extra,
    }
    return render(request, template, **context)


@app.get("/partials/atelier", response_class=HTMLResponse)
def atelier_fragment(request: Request):
    """État de l'atelier, relu par la barre latérale toutes les dix secondes."""
    return render(request, "partials/atelier.html", atelier=atelier.status(workspace()))


# --- Bibliothèque -------------------------------------------------------------------
def library_card(name: str, project: Project) -> dict[str, object]:
    status = project.status()
    running = queue.active(name)
    last = next(iter(queue.list(name, limit=1)), None)
    tracks = project.tracks()
    seconds = sum(float(t["seconds"]) for t in tracks)

    if running:
        label, tone = f"{running.kind} · {running.progress:.0%}", "accent"
    elif status["m4b"]:
        label, tone = "M4B prêt", "ok"
    elif last and last.state is State.FAILED:
        label, tone = "dernière tâche en échec", "bad"
    elif tracks and all(t["current"] for t in tracks):
        label, tone = "synthétisé · à assembler", "ok"
    elif project.pending_review:
        label, tone = f"à relire · {project.pending_review} chapitre(s)", "warn"
    elif status["segments"]:
        label, tone = "préparé · à synthétiser", ""
    else:
        label, tone = "importé", ""

    return {
        "name": name,
        "project": project,
        "chapters": status["chapitres"],
        "seconds": seconds,
        "label": label,
        "tone": tone,
        "progress": running.progress if running else None,
        "m4b": status["m4b"],
        "kind": project.kind or (Path(project.source).suffix.lstrip(".") if project.source else ""),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    cards = [library_card(name, project) for name, project in all_projects()]
    total = sum(float(c["seconds"]) for c in cards)
    return shell(request, "index.html", cards=cards, total_seconds=total, nav="library")


@app.post("/peek")
async def peek_book(file: UploadFile):
    """Ce que le fichier dit de lui-même, pour préremplir le formulaire de dépôt."""
    import tempfile

    from ..ingest.metadata import peek

    suffix = Path(file.filename or "").suffix.lower() or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        shutil.copyfileobj(file.file, handle)
        temporary = Path(handle.name)
    try:
        return peek(temporary)
    finally:
        temporary.unlink(missing_ok=True)


@app.post("/projects")
async def create_project(
    file: UploadFile,
    title: str = Form(""),
    author: str = Form(""),
    language: str = Form(""),
):
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
    if language:
        document.language = language

    name = slug(document.title)
    directory = root / name
    suffix = 2
    while directory.exists():
        directory = root / f"{name}-{suffix}"
        suffix += 1

    Project.create(directory, document)
    return RedirectResponse(f"/projects/{directory.name}", status_code=303)


@app.post("/projects/{name}/delete")
def delete_project(name: str):
    """Supprime un projet et tout ce qu'il contient.

    La suppression emporte le texte relu et les heures de synthèse — un livre de
    quatre-vingt-dix minutes en demande vingt à produire. Une tâche en cours écrirait
    d'ailleurs dans un dossier disparu : on refuse tant qu'elle n'est pas finie.
    """
    directory = project_dir(name)
    if queue.active(name):
        raise HTTPException(409, "Une tâche est en cours sur ce projet : attendez sa fin.")
    shutil.rmtree(directory)
    return RedirectResponse("/", status_code=303)


# --- Chapitres et métadonnées -------------------------------------------------------
@app.get("/projects/{name}", response_class=HTMLResponse)
def show_project(request: Request, name: str, cover: str = ""):
    project = load_project(name)
    return shell(
        request,
        "project/chapters.html",
        name,
        "chapters",
        cover_notice=cover,
        tracks=project.tracks(),
        status=project.status(),
        has_cover=project.cover_source is not None,
    )


@app.post("/projects/{name}/meta")
def save_meta(
    name: str,
    title: str = Form(""),
    author: str = Form(""),
    year: str = Form(""),
    language: str = Form(""),
):
    project = load_project(name)
    project.title = title.strip() or project.title
    project.author = author.strip() or "Inconnu"
    project.year = year.strip() or None
    project.language = language.strip() or project.language
    project.save()
    return RedirectResponse(f"/projects/{name}", status_code=303)


@app.post("/projects/{name}/cover")
async def upload_cover(name: str, file: UploadFile):
    """Remplace la couverture. L'image d'origine du livre reste le repli."""
    project = load_project(name)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(400, "Couverture : JPEG ou PNG attendu.")
    for old in project.root.glob("cover.*"):
        old.unlink()
    with (project.root / f"cover{suffix}").open("wb") as target:
        shutil.copyfileobj(file.file, target)
    return RedirectResponse(f"/projects/{name}", status_code=303)


@app.post("/projects/{name}/cover/search")
def search_cover_online(name: str):
    """Cherche la couverture sur Open Library, d'après le titre et l'auteur."""
    from ..ingest.metadata import search_cover

    project = load_project(name)
    data = search_cover(project.title, project.author if project.author != "Inconnu" else "")
    if not data:
        return RedirectResponse(f"/projects/{name}?cover=introuvable", status_code=303)
    for old in project.root.glob("cover.*"):
        old.unlink()
    (project.root / "cover.jpg").write_bytes(data)
    return RedirectResponse(f"/projects/{name}?cover=trouvee", status_code=303)


@app.get("/projects/{name}/cover")
def show_cover(name: str):
    """La couverture telle que le M4B la portera : déposée, ou extraite du livre."""
    from ..assemble import extract_cover

    project = load_project(name)
    if project.cover:
        return FileResponse(project.cover)
    cache = project.root / "work" / "cover.jpg"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            found = extract_cover(project.cover_source, cache)
        except Exception:
            found = None
        if not found:
            raise HTTPException(404, "Pas de couverture")
    return FileResponse(cache)


@app.post("/projects/{name}/chapters/{number}/{action}")
async def edit_chapter(request: Request, name: str, number: int, action: str):
    """Renomme un chapitre, le retire, ou le recolle au précédent."""
    if action not in {"delete", "merge", "rename"}:
        raise HTTPException(404, "Action inconnue")
    project = load_project(name)
    if action != "rename" and queue.active(name):
        raise HTTPException(409, "Une tâche est en cours sur ce projet : attendez sa fin.")
    try:
        if action == "rename":
            form = await request.form()
            project.set_chapter_title(number, str(form.get("title", "")))
        elif action == "merge":
            project.merge_into_previous(number)
        else:
            project.delete_chapter(number)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return RedirectResponse(f"/projects/{name}", status_code=303)


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


def page_size(project: Project, chapter: Chapter) -> tuple[int, int] | None:
    """Proportions de la première page du chapitre, pour dimensionner le cadre."""
    if pages.kind(project) != "epub" or not chapter.source_pages:
        return None
    try:
        book = pages.Pages(project.source)
        return pages.size(book.raw(chapter.source_pages[0] - 1))
    except (OSError, IndexError, KeyError):
        return None


@app.get("/projects/{name}/review/{number}", response_class=HTMLResponse)
def review_chapter(request: Request, name: str, number: int, find: str = ""):
    """Éditeur de relecture : le texte à corriger, la page d'origine en regard."""
    from ..review import inspect_texts

    project = load_project(name)
    path = project.text_dir / f"ch{number:02d}.md"
    if not path.exists():
        raise HTTPException(404, "Chapitre introuvable")

    chapter = Chapter.from_markdown(path.read_text(encoding="utf-8"))
    suspects = inspect_texts({path.name: chapter.text}, language=project.language)
    reasons = sorted({s.reason for s in suspects})
    numbers = sorted(int(p.stem.removeprefix("ch")) for p in project.text_dir.glob("ch*.md"))
    titles = {int(s["number"]): s["title"] for s in project.chapter_states()}
    return shell(
        request,
        "project/review.html",
        name,
        "review",
        chapter=chapter,
        suspects=suspects,
        reasons=reasons,
        personas=project.personas(),
        suggestions=[s for s in load_suggestions(project) if s.chapter == path.name],
        numbers=numbers,
        titles=titles,
        previous=max([n for n in numbers if n < number], default=None),
        following=min([n for n in numbers if n > number], default=None),
        find=find,
        page_kind=pages.kind(project),
        page_size=page_size(project, chapter),
        llm_enabled=setting("VOXLIBRIS_LLM_ENABLED", "1") != "0",
        llm_model=setting("VOXLIBRIS_LLM_MODEL"),
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
    target = project.clean_dir / f"ch{number:02d}.md"
    content = chapter.to_markdown()
    # Enregistrer sans rien changer ne date pas le fichier : la date du texte sert à
    # signaler une correction faite après la préparation, pas un simple passage.
    if not target.exists() or target.read_text(encoding="utf-8") != content:
        target.write_text(content, encoding="utf-8")
    return RedirectResponse(f"/projects/{name}/review/{number}", status_code=303)


@app.get("/projects/{name}/page/{number}")
def chapter_page(name: str, number: int, page: int = 0):
    """Une page d'origine du chapitre, pour la relecture côte à côte.

    Image rendue pour un PDF ; page HTML servie telle quelle, sans script, pour un EPUB
    paginé.
    """
    project = load_project(name)
    source = pages.kind(project)
    if source is None:
        raise HTTPException(404, "Pas de source paginée pour ce projet")

    chapter = Chapter.from_markdown(
        (project.text_dir / f"ch{number:02d}.md").read_text(encoding="utf-8")
    )
    if not chapter.source_pages:
        raise HTTPException(404, "Chapitre sans pages d'origine")

    first, last = chapter.source_pages
    wanted = min(max(first + page, first), last)
    if source == "epub":
        book = pages.Pages(project.source)
        if wanted > len(book):
            raise HTTPException(404, "Page hors du livre")
        return HTMLResponse(
            book.page(wanted - 1, f"/projects/{name}/source"),
            headers={"Content-Security-Policy": pages.CSP},
        )

    cache = project.root / "work" / "pages"
    cache.mkdir(parents=True, exist_ok=True)
    image = cache / f"p{wanted:04d}.png"
    if not image.exists():
        import pymupdf

        pymupdf.open(project.source)[wanted - 1].get_pixmap(dpi=150).save(image)
    return FileResponse(image, media_type="image/png")


@app.get("/projects/{name}/source/{path:path}")
def source_asset(name: str, path: str):
    """Police, image ou feuille de style d'un EPUB paginé, pour afficher ses pages."""
    project = load_project(name)
    if pages.kind(project) != "epub":
        raise HTTPException(404, "Pas de ressources pour ce projet")
    found = pages.Pages(project.source).asset(path)
    if found is None:
        raise HTTPException(404, "Ressource introuvable")
    data, media = found
    return Response(data, media_type=media, headers={"Cache-Control": "max-age=86400"})


# --- Préparation --------------------------------------------------------------------
@app.get("/projects/{name}/prepare", response_class=HTMLResponse)
def prepare_page(request: Request, name: str):
    import json

    project = load_project(name)
    preview: list[dict] = []
    first = next(iter(sorted(project.segments_dir.glob("ch*.jsonl"))), None)
    if first:
        with first.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    preview.append(json.loads(line))
                if len(preview) >= 8:
                    break
    roles = project.roles()
    return shell(
        request,
        "project/prepare.html",
        name,
        "prepare",
        status=project.status(),
        tracks=project.tracks(),
        preview=preview,
        dialogues=roles.get("dialogue", 0),
        personas=[(p, roles[p]) for p in project.personas() if roles.get(p)],
    )


# --- Voix ---------------------------------------------------------------------------
@app.get("/projects/{name}/voices", response_class=HTMLResponse)
def voices_page(request: Request, name: str, cloning: str = ""):
    project = load_project(name)
    pulse = atelier.status(workspace())
    available = pulse.get("engines") or {}
    known = catalogues(project.language)
    return shell(
        request,
        "project/voices.html",
        name,
        "voices",
        engines=ENGINES,
        available=available,
        served=pulse.get("served") or {},
        catalogues=known,
        candidates=sample_candidates(project.language, project, known),
        samples=samples_of(project),
        voxtral_local=voxtral_is_local(),
        voice_samples=voice_samples(),
        cloning=cloning,
        status=project.status(),
        characters=sum(len(t) for t in project.chapter_texts().values()),
    )


@app.post("/projects/{name}/voices/clone")
async def upload_voice(name: str, file: UploadFile, label: str = Form("")):
    """Dépose un extrait de voix à cloner. Le fichier est la voix : son nom, l'intitulé.

    Les serveurs ZONOS2 et OmniVoice relisent le dossier à chaque liste : rien à
    relancer. L'extrait n'est
    lié à aucun projet — une voix sert à tous les livres.
    """
    from ..tts.zonos2 import AUDIO_EXTENSIONS

    load_project(name)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(400, "Extrait de voix : " + ", ".join(AUDIO_EXTENSIONS) + ".")
    stem = slug(label or Path(file.filename or "").stem)
    if stem == "livre":
        raise HTTPException(400, "Donnez un nom à la voix.")
    folder = voices_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{stem}.*"):
        old.unlink()
    with (folder / f"{stem}{suffix}").open("wb") as target:
        shutil.copyfileobj(file.file, target)
    forget_catalogues()
    return RedirectResponse(f"/projects/{name}/voices?cloning=deposee", status_code=303)


@app.post("/projects/{name}/voices/clone/delete")
def delete_voice(name: str, sample: str = Form(...)):
    """Retire un extrait : la voix disparaît du serveur à sa prochaine liste.

    Le champ ne s'appelle pas « file » : le bouton vit dans le formulaire de dépôt, dont
    le sélecteur de fichier porte déjà ce nom, et un fichier vide l'emporterait.
    """
    load_project(name)
    target = voices_dir() / Path(sample).name
    if target.is_file():
        target.unlink()
    forget_catalogues()
    return RedirectResponse(f"/projects/{name}/voices", status_code=303)


@app.post("/projects/{name}/voice")
def choose_voice(name: str, backend: str = Form(...), voice: str = Form("")):
    """Retient un moteur et une voix. La synthèse s'en servira, le M4B le gravera."""
    if backend not in BACKENDS:
        raise HTTPException(400, "Moteur inconnu")
    project = load_project(name)
    # Les voix de la distribution appartiennent au moteur : un autre moteur ne les
    # connaît pas. Les personas, eux, restent — ils appartiennent au texte.
    if backend != project.backend:
        project.voices = {persona: "" for persona in project.voices}
    project.backend = backend
    project.voice = voice.strip() or None
    project.save()
    return RedirectResponse(f"/projects/{name}/synth", status_code=303)


@app.post("/projects/{name}/cast")
async def choose_cast(request: Request, name: str):
    """Retient la distribution : pour chaque persona, sa voix — vide, le narrateur le lit.

    Les personas viennent du texte (marqueurs « @Charles ») et de la règle des
    répliques ; on peut en nommer un nouveau ici avant de lui attribuer des paragraphes
    dans la Relecture.
    """
    project = load_project(name)
    form = await request.form()
    voices: dict[str, str] = {}
    for persona, voice in zip(form.getlist("persona"), form.getlist("voice"), strict=False):
        persona = str(persona).strip().lstrip("@")
        if persona and persona.casefold() != "narrateur":
            voices[persona] = str(voice).strip()
    project.voices = voices
    project.save()
    return RedirectResponse(f"/projects/{name}/synth", status_code=303)


# --- Synthèse -----------------------------------------------------------------------
@app.get("/projects/{name}/synth", response_class=HTMLResponse)
def synth_page(request: Request, name: str, chapter: int = 0, cause: str = ""):
    project = load_project(name)
    flagged = project.flagged_segments()
    approved = project.approved_segments()
    causes = sorted({str(f.get("cause") or "") for f in flagged} - {""})
    if approved:
        causes.append(APPROVED)
    if cause == APPROVED:
        flagged = approved
    elif cause:
        flagged = [f for f in flagged if f.get("cause") == cause]
    if chapter:
        flagged = [f for f in flagged if f["chapter"] == chapter]
    return shell(
        request,
        "project/synth.html",
        name,
        "synth",
        tracks=project.tracks(),
        flagged=flagged,
        causes=causes,
        filter_chapter=chapter,
        filter_cause=cause,
        approved_count=len(approved),
        approved_label=APPROVED,
        status=project.status(),
        engines=ENGINES,
        catalogues=catalogues(project.language),
        supports_speed={
            k: (voxtral_is_local() if k == "voxtral" else cls.supports_speed)
            for k, cls in BACKENDS.items()
        },
        speed_range=SPEED_RANGE,
        voxtral_local=voxtral_is_local(),
        jobs=queue.list(name, limit=6),
        personas=project.personas(),
        roles=project.roles(),
    )


# Pseudo-cause du filtre : montrer les segments validés, pour revenir dessus.
APPROVED = "validé"


@app.post("/projects/{name}/segments/approve")
def approve_segment(
    name: str,
    chapter: int = Form(...),
    idx: int = Form(...),
    undo: str = Form(""),
    back: str = Form(""),
):
    """Valide un segment à l'oreille, ou annule cette validation."""
    project = load_project(name)
    if not project.approve_segment(chapter, idx, approved=not undo):
        raise HTTPException(404, "Segment introuvable")
    target = back if back.startswith(f"/projects/{name}/synth") else f"/projects/{name}/synth"
    return RedirectResponse(target, status_code=303)


# --- Assemblage ---------------------------------------------------------------------
@app.get("/projects/{name}/assemble", response_class=HTMLResponse)
def assemble_page(request: Request, name: str):
    from ..assemble import LOUDNESS_LUFS

    project = load_project(name)
    mp3s = sorted((project.out_dir / "mp3").glob("*.mp3"))
    return shell(
        request,
        "project/assemble.html",
        name,
        "assemble",
        tracks=project.tracks(),
        status=project.status(),
        mp3s=[f"mp3/{p.name}" for p in mp3s],
        loudness=LOUDNESS_LUFS,
        has_cover=project.cover_source is not None,
        jobs=queue.list(name, limit=6),
    )


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
    project = load_project(name)
    if queue.active(name):
        raise HTTPException(409, "Une tâche est déjà en cours pour ce projet")

    form = await request.form()
    params: dict = {}
    if kind == "synth":
        params = {
            "backend": str(form.get("backend") or project.backend or "xtts"),
            "voice": str(form.get("voice") or "") or None,
            "force": bool(form.get("force")),
            "speed": _number(form.get("speed"), *SPEED_RANGE),
        }
        if device := str(form.get("device") or ""):
            params["device"] = device
    elif kind == "resynth":
        params = {"chapter": int(str(form.get("chapter"))), "idx": int(str(form.get("idx")))}
    elif kind == "normalize":
        params = {
            "pause_scale": _number(form.get("pause_scale"), 0.5, 3.0),
            "announce": bool(form.get("announce")),
        }
    elif kind == "sample":
        picks = [str(v) for v in form.getlist("voice")]
        if extra := str(form.get("extra_voice") or "").strip():
            picks.append(f"{form.get('extra_backend', 'xtts')}/{extra}")
        params = {
            "voices": [
                {"backend": b, "voice": v} for b, _, v in (pick.partition("/") for pick in picks)
            ]
        }
    elif kind == "assemble":
        params = {"skip_mp3": not form.get("mp3")}
    elif kind != "proofread":
        raise HTTPException(404, "Tâche inconnue")

    queue.enqueue(name, kind, **params)
    return RedirectResponse(
        f"/projects/{name}/{LANDING.get(kind, '')}".rstrip("/"), status_code=303
    )


@app.post("/projects/{name}/jobs/{job_id}/cancel")
def cancel_job(request: Request, name: str, job_id: int):
    """Arrête une tâche. En cours, elle s'interrompt au prochain segment."""
    load_project(name)
    try:
        job = queue.get(job_id)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    if job.project != name:
        raise HTTPException(404, "Tâche inconnue pour ce projet")
    queue.request_cancel(job_id)
    back = request.headers.get("referer") or f"/projects/{name}/synth"
    return RedirectResponse(back, status_code=303)


@app.get("/projects/{name}/progress", response_class=HTMLResponse)
def job_progress(request: Request, name: str):
    """Fragment rafraîchi par sondage : état de la tâche courante."""
    load_project(name)
    return render(
        request,
        "partials/progress.html",
        name=name,
        job=queue.active(name),
        jobs=queue.list(name, limit=1),
    )


@app.get("/projects/{name}/jobs", response_class=HTMLResponse)
def project_jobs(request: Request, name: str):
    return shell(request, "jobs.html", name, "", jobs=queue.list(name, limit=50), scope=name)


@app.get("/jobs", response_class=HTMLResponse)
def all_jobs(request: Request):
    return shell(request, "jobs.html", jobs=queue.list(limit=80), scope="", nav="jobs")


# --- Fichiers produits --------------------------------------------------------------
@app.get("/projects/{name}/files/{path:path}")
def download(name: str, path: str):
    project = load_project(name)
    target = (project.out_dir / path).resolve()
    if not target.is_relative_to(project.out_dir.resolve()) or not target.is_file():
        raise HTTPException(404, "Fichier introuvable")
    # Une piste WAV se lit dans la page ; un livre audio se télécharge.
    inline = target.suffix.lower() == ".wav"
    return FileResponse(
        target, filename=target.name, content_disposition_type="inline" if inline else "attachment"
    )


@app.get("/api/voices/{backend}")
def list_voices(backend: str):
    """Voix d'un moteur. Sollicité par l'interface au changement de moteur."""
    if backend not in BACKENDS:
        return {"voices": [], "error": "moteur inconnu"}
    known = catalogues().get(backend) or []
    if known:
        return {"voices": known}
    from ..tts.backends import load

    try:
        return {"voices": load(backend).voices()}
    except Exception as error:
        # Un moteur non installé ne doit pas casser la page : on le dit, simplement.
        return {"voices": [], "error": str(error)}


# --- Réglages -----------------------------------------------------------------------
def settings_view() -> dict[str, dict[str, str]]:
    """Chaque réglage modifiable, sa valeur affichable et sa provenance."""
    view = {}
    for key in EDITABLE:
        value = setting(key)
        shown = ("•" * 24 if value else "") if key in SECRETS else value
        view[key] = {"value": shown, "set": bool(value), "origin": origin(key)}
    return view


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, saved: int = 0, released: int = 0):
    return shell(
        request,
        "settings.html",
        settings=settings_view(),
        saved=bool(saved),
        released=bool(released),
        data_dir=str(workspace()),
        nav="settings",
    )


@app.post("/settings")
async def save_settings(request: Request):
    """Enregistre les réglages. Un secret laissé masqué n'est pas réécrit."""
    form = await request.form()
    values: dict[str, str] = {}
    for key in EDITABLE:
        if key not in form:
            continue
        value = str(form.get(key) or "").strip()
        if key in SECRETS and value and set(value) == {"•"}:
            continue
        values[key] = value
    # Une case à cocher absente du formulaire vaut « non ».
    for flag, on in (("COQUI_TOS_AGREED", "1"), ("VOXLIBRIS_LLM_ENABLED", "1")):
        if flag in form or f"{flag}__present" in form:
            values[flag] = (
                on if form.get(flag) else ("0" if flag == "VOXLIBRIS_LLM_ENABLED" else "")
            )
    write_settings(values)
    return RedirectResponse("/settings?saved=1", status_code=303)


@app.post("/atelier/release")
def release_gpu():
    """Rend la carte graphique tout de suite : les serveurs de synthèse sont endormis.

    C'est une tâche comme une autre, exécutée par l'atelier — seul lui parle à Docker —
    et qui attend donc son tour derrière une synthèse en cours.
    """
    if queue.active("atelier") is None:
        queue.enqueue("atelier", "release")
    return RedirectResponse("/settings?released=1", status_code=303)


@app.post("/settings/test/{service}", response_class=HTMLResponse)
def test_service(request: Request, service: str):
    """Sonde un service et rend le verdict en un fragment."""
    if service == "llm":
        from ..llm import LLM, LLMConfig

        try:
            verdict = LLM(LLMConfig.from_env()).probe()
            ok, text = True, str(verdict) if verdict else "le modèle répond"
        except Exception as error:
            ok, text = False, str(error)
    elif service == "voxtral":
        from ..tts.voxtral import Client

        try:
            client = Client()
            if client.is_local:
                client.probe()
            voices = client.voices()
            ok = True
            text = f"{'serveur local' if client.is_local else 'API Mistral'} · {len(voices)} voix"
        except Exception as error:
            ok, text = False, str(error)
    elif service == "zonos2":
        from ..tts.zonos2 import Client

        try:
            client = Client()
            client.probe()
            voices = client.speakers()
            ok = True
            text = f"serveur local · {len(voices)} voix"
        except Exception as error:
            ok, text = False, str(error)
    elif service == "omnivoice":
        from ..tts.omnivoice import Client

        try:
            client = Client()
            health = client.probe()
            voices = client.speakers()
            ok = True
            text = f"serveur local · {health.get('device') or '?'} · {len(voices)} voix"
        except Exception as error:
            ok, text = False, str(error)
    else:
        raise HTTPException(404, "Service inconnu")
    return render(request, "partials/verdict.html", ok=ok, text=text)
