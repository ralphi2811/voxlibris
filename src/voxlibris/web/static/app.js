// L'Atelier, côté navigateur : quelques comportements, sans bibliothèque.
//
// Le rendu reste fait par le serveur. Ce qui vit ici tient en cinq idées : relire un
// fragment à intervalle régulier (l'avancement d'une tâche, l'état de l'atelier), demander
// confirmation avant un geste irréversible, faire suivre une glissière par son affichage,
// aider la relecture (localiser, remplacer, accepter une proposition), et jouer un
// segment isolé d'une piste à partir de son instant.

(function () {
  "use strict";

  // --- Fragments relus périodiquement --------------------------------------------------
  //   <div data-poll="/chemin" data-interval="2000">…</div>
  // Le sondage s'arrête de lui-même quand le fragment ne contient plus de tâche en cours,
  // sauf s'il porte data-forever (l'état de l'atelier, qui change à tout moment).
  function isRunning(node) { return !!node.querySelector(".job.running"); }

  function refresh(node) {
    var wasRunning = isRunning(node);
    fetch(node.dataset.poll, { headers: { "X-Requested-With": "fetch" } })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
      .then(function (html) {
        node.innerHTML = html;
        if ("forever" in node.dataset) return;
        if (!isRunning(node)) {
          stop(node);
          // La fin d'une tâche débloque les étapes suivantes, change des compteurs, fait
          // apparaître un téléchargement : tout cela a été rendu avant, et est périmé.
          if (wasRunning) window.location.reload();
        }
      })
      .catch(function () { if (!("forever" in node.dataset)) stop(node); });
  }
  function stop(node) { if (node._timer) { clearInterval(node._timer); node._timer = null; } }
  function start(node) {
    var interval = parseInt(node.dataset.interval || "2000", 10);
    node._timer = setInterval(function () { refresh(node); }, interval);
    if (!("forever" in node.dataset)) refresh(node);
  }

  // --- Confirmation avant un geste irréversible ----------------------------------------
  function guard(form) {
    form.addEventListener("submit", function (e) { if (!window.confirm(form.dataset.confirm)) e.preventDefault(); });
  }

  // --- Glissières et champs qui se répondent -------------------------------------------
  function bindRange(input) {
    var out = document.getElementById(input.dataset.out);
    if (!out) return;
    var show = function () { out.textContent = Number(input.value).toFixed(2) + (input.dataset.suffix || ""); };
    input.addEventListener("input", show);
    show();
  }

  // Un formulaire de filtre part tout seul quand on change un de ses sélecteurs.
  function autosubmit(form) {
    form.querySelectorAll("select").forEach(function (s) { s.addEventListener("change", function () { form.submit(); }); });
  }

  // Un champ de fichier marqué data-submit envoie son formulaire dès qu'un fichier est choisi.
  function submitOnPick(input) {
    input.addEventListener("change", function () { if (input.files.length) input.form.submit(); });
  }

  // Le champ de titre d'un chapitre : le bouton n'apparaît que si le titre a changé.
  function inlineEdit(form) {
    var input = form.querySelector("input[name=title]"), save = form.querySelector(".save");
    if (!input || !save) return;
    input.addEventListener("input", function () { save.hidden = input.value.trim() === input.dataset.original; });
  }

  // --- Bibliothèque ------------------------------------------------------------------------
  function libraryFilter(box) {
    var cards = document.querySelectorAll("#books .book");
    box.addEventListener("input", function () {
      var q = box.value.trim().toLowerCase();
      cards.forEach(function (c) { c.hidden = q && c.dataset.search.indexOf(q) < 0; });
    });
  }
  function dropzone(form) {
    var input = form.querySelector("input[type=file]"), label = document.getElementById("file-label");
    ["dragenter", "dragover"].forEach(function (ev) { form.addEventListener(ev, function (e) { e.preventDefault(); form.classList.add("over"); }); });
    ["dragleave", "drop"].forEach(function (ev) { form.addEventListener(ev, function (e) { e.preventDefault(); form.classList.remove("over"); }); });
    var chosen = function () {
      if (!input.files.length) return;
      if (label) label.textContent = input.files[0].name;
      prefill(form, input.files[0]);
    };
    form.addEventListener("drop", function (e) {
      if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; chosen(); }
    });
    if (input) input.addEventListener("change", chosen);
  }
  // Le livre dit son titre, son auteur, sa langue : on les propose, sans écraser ce
  // que l'utilisateur a tapé lui-même. Un champ rempli par nous est marqué, pour être
  // remplacé si un autre fichier est choisi ensuite.
  function prefill(form, file) {
    var data = new FormData(); data.append("file", file);
    fetch("/peek", { method: "POST", body: data }).then(function (r) { return r.ok ? r.json() : {}; }).then(function (meta) {
      // Ce qu'un fichier précédent avait proposé ne vaut plus pour celui-ci.
      form.querySelectorAll("[data-auto]").forEach(function (field) {
        if (field.tagName === "SELECT") field.selectedIndex = 0; else field.value = "";
        delete field.dataset.auto;
      });
      ["title", "author", "language"].forEach(function (key) {
        var field = form.querySelector("[name=" + key + "]");
        if (!field || !meta[key]) return;
        if (field.value && field.dataset.auto !== "1" && field.tagName !== "SELECT") return;
        if (field.tagName === "SELECT") {
          if (!field.dataset.auto && field.dataset.touched) return;
          if ([].some.call(field.options, function (o) { return o.value === meta[key]; })) { field.value = meta[key]; field.dataset.auto = "1"; }
          return;
        }
        field.value = meta[key]; field.dataset.auto = "1";
      });
    }).catch(function () { /* préremplir est une commodité : en cas d'échec, on laisse les champs */ });
    form.querySelectorAll("input[name=title], input[name=author]").forEach(function (field) {
      field.addEventListener("input", function () { delete field.dataset.auto; }, { once: true });
    });
    var select = form.querySelector("select[name=language]");
    if (select) select.addEventListener("change", function () { select.dataset.touched = "1"; delete select.dataset.auto; });
  }

  // --- Voix : la liste des voix suit le moteur -----------------------------------------------
  function voiceList(select, datalistId) {
    var list = document.getElementById(datalistId);
    if (!list) return;
    var fill = function () {
      fetch("/api/voices/" + select.value).then(function (r) { return r.json(); }).then(function (data) {
        list.innerHTML = "";
        (data.voices || []).forEach(function (v) { var o = document.createElement("option"); o.value = v; list.appendChild(o); });
      }).catch(function () {});
    };
    select.addEventListener("change", fill);
    fill();
  }
  function enginePicker(box) {
    var radios = box.querySelectorAll("input[type=radio]"), list = document.getElementById("synth-voices"),
        hint = document.getElementById("speed-hint");
    // Les moteurs sans réglage de débit : dits par le serveur, car Voxtral en a un
    // quand il est servi en local et pas par l'API.
    var noSpeed = {};
    ((hint && hint.dataset.nospeed) || "").split(/\s+/).forEach(function (k) { if (k) noSpeed[k] = true; });
    var fill = function () {
      var chosen = box.querySelector("input:checked");
      if (!chosen) return;
      if (hint) hint.textContent = noSpeed[chosen.value]
        ? "Ce moteur ne module pas le débit : la vitesse est sans effet."
        : "Changer la vitesse refait tout le livre, pour un débit constant.";
      if (!list) return;
      fetch("/api/voices/" + chosen.value).then(function (r) { return r.json(); }).then(function (data) {
        list.innerHTML = "";
        (data.voices || []).forEach(function (v) { var o = document.createElement("option"); o.value = v; list.appendChild(o); });
      }).catch(function () {});
    };
    radios.forEach(function (r) { r.addEventListener("change", fill); });
    fill();
  }

  // --- Sélecteur de fichier habillé : montrer ce qui a été choisi ----------------------
  document.querySelectorAll(".file-pick input[type=file]").forEach(function (input) {
    input.addEventListener("change", function () {
      var label = input.closest(".file-pick"), name = label.querySelector(".file-name");
      if (input.files && input.files[0]) { name.textContent = input.files[0].name; label.classList.add("picked"); }
    });
  });

  // --- Synthèse : écouter une piste ou un segment isolé ------------------------------------
  // La piste entière est chargée par le navigateur, qui saute à l'instant voulu ; on
  // arrête la lecture à la fin du segment. Aucun découpage côté serveur. Le lecteur
  // apparaît au premier clic et dit ce qui joue.
  function listen(button) {
    var player = document.getElementById("player"), bar = document.getElementById("player-bar");
    if (!player) return;
    var stop = function () {
      document.querySelectorAll(".listen.playing").forEach(function (b) { b.classList.remove("playing"); });
    };
    player.addEventListener("pause", stop);
    player.addEventListener("ended", stop);
    button.addEventListener("click", function () {
      if (button.classList.contains("playing")) { player.pause(); return; }
      var start = Number(button.dataset.start || 0), end = button.dataset.end ? Number(button.dataset.end) : null;
      if (player.dataset.src !== button.dataset.src) { player.src = button.dataset.src; player.dataset.src = button.dataset.src; }
      if (bar) { bar.hidden = false; document.getElementById("player-label").textContent = button.dataset.label || ""; }
      player.currentTime = start;
      player.play();
      stop();
      button.classList.add("playing");
      if (end === null) return;
      var onTime = function () {
        if (player.currentTime >= end + 0.05) { player.pause(); player.removeEventListener("timeupdate", onTime); }
      };
      player.addEventListener("timeupdate", onTime);
    });
  }

  // --- Relecture -----------------------------------------------------------------------------
  function review() {
    var area = document.querySelector("textarea[name=text]");
    if (!area) return;

    var select = function (at, length) {
      area.focus();
      area.setSelectionRange(at, at + length);
      var ratio = at / Math.max(area.value.length, 1);
      area.scrollTop = ratio * area.scrollHeight - area.clientHeight / 2;
    };

    // Un mot signalé se sélectionne dans le texte : c'est le va-et-vient qui rend la
    // relecture d'un scan supportable.
    document.querySelectorAll(".locate").forEach(function (b) {
      b.addEventListener("click", function () {
        var at = area.value.indexOf(b.dataset.word);
        if (at >= 0) select(at, b.dataset.word.length);
      });
    });

    // Accepter une proposition : la substitution est faite à la position exacte relevée
    // lors de l'analyse, et seulement si le texte s'y trouve encore tel quel. Rien n'est
    // envoyé : c'est « Enregistrer » qui décide, comme pour le reste.
    var accept = function (b) {
      var word = b.dataset.word, replacement = b.dataset.replacement, offset = Number(b.dataset.offset);
      var at = area.value.startsWith(word, offset) ? offset : area.value.indexOf(word);
      var card = b.closest(".proposal");
      if (at < 0) { b.disabled = true; b.title = "cette forme n'est plus dans le texte"; return false; }
      area.value = area.value.slice(0, at) + replacement + area.value.slice(at + word.length);
      select(at, replacement.length);
      if (card) card.classList.add("applied");
      b.disabled = true;
      return true;
    };
    document.querySelectorAll(".accept").forEach(function (b) { b.addEventListener("click", function () { accept(b); }); });
    document.querySelectorAll(".dismiss").forEach(function (b) {
      b.addEventListener("click", function () { var card = b.closest(".proposal"); if (card) card.hidden = true; });
    });
    var all = document.getElementById("accept-all"), none = document.getElementById("dismiss-all");
    // Les positions sont relevées sur le texte d'origine : on applique de la fin vers le
    // début pour qu'aucune substitution ne décale celles qui la suivent.
    if (all) all.addEventListener("click", function () {
      var buttons = Array.prototype.slice.call(document.querySelectorAll(".accept:not(:disabled)"));
      buttons.sort(function (a, b) { return Number(b.dataset.offset) - Number(a.dataset.offset); }).forEach(accept);
    });
    if (none) none.addEventListener("click", function () { document.querySelectorAll(".proposal[data-reason='modèle']").forEach(function (c) { c.hidden = true; }); });

    // Filtrer les propositions par cause.
    var filter = document.getElementById("reason-filter"), count = document.getElementById("suspect-count");
    if (filter) filter.querySelectorAll(".chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        filter.querySelectorAll(".chip").forEach(function (c) { c.classList.remove("on", "accent"); c.classList.add("muted"); });
        chip.classList.add("on", "accent"); chip.classList.remove("muted");
        var wanted = chip.dataset.reason, shown = 0;
        document.querySelectorAll(".proposal").forEach(function (p) { p.hidden = wanted && p.dataset.reason !== wanted; if (!p.hidden) shown++; });
        if (count) count.textContent = shown;
      });
    });

    // Rechercher et remplacer, dans la zone de saisie seulement.
    var find = document.getElementById("find"), replace = document.getElementById("replace");
    var locateNext = function () {
      var q = find.value; if (!q) return -1;
      var from = area.selectionEnd || 0;
      var at = area.value.indexOf(q, from);
      if (at < 0) at = area.value.indexOf(q);
      if (at >= 0) select(at, q.length);
      return at;
    };
    if (find) {
      find.addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); locateNext(); } });
      if (find.value) setTimeout(locateNext, 50);
    }
    var one = document.getElementById("replace-one"), every = document.getElementById("replace-all");
    if (one) one.addEventListener("click", function () {
      var q = find.value; if (!q) return;
      var at = area.selectionStart;
      if (area.value.substr(at, q.length) !== q) at = locateNext();
      if (at < 0) return;
      area.value = area.value.slice(0, at) + replace.value + area.value.slice(at + q.length);
      select(at, replace.value.length);
    });
    if (every) every.addEventListener("click", function () {
      var q = find.value; if (!q) return;
      var n = area.value.split(q).length - 1;
      if (n && window.confirm("Remplacer " + n + " occurrence(s) de « " + q + " » ?")) area.value = area.value.split(q).join(replace.value);
    });

    // Feuilleter les pages d'origine : une image pour un PDF, un cadre pour un EPUB paginé.
    var image = document.getElementById("page-image");
    if (image) {
      var offset = 0, total = Number(image.dataset.total), first = Number(image.dataset.first);
      var label = document.getElementById("page-label");
      var show = function () { image.src = image.dataset.url + "?page=" + offset; if (label) label.textContent = first + offset; };
      document.getElementById("page-prev").onclick = function () { offset = Math.max(0, offset - 1); show(); };
      document.getElementById("page-next").onclick = function () { offset = Math.min(total - 1, offset + 1); show(); };
      if (image.tagName === "IFRAME") fitPage(image);
    }

    var jump = document.getElementById("chapter-jump");
    if (jump) jump.addEventListener("change", function () { window.location = jump.value; });
  }

  // Une page d'EPUB paginé est servie à sa taille réelle, sans script : c'est ici qu'on
  // la réduit à la largeur du volet, en gardant ses proportions.
  function fitPage(frame) {
    var box = frame.parentNode;
    var fit = function () {
      var w = Number(frame.dataset.width), h = Number(frame.dataset.height);
      if (!w || !h) return;
      box.style.aspectRatio = w + " / " + h;
      frame.style.width = w + "px"; frame.style.height = h + "px";
      frame.style.transform = "scale(" + box.clientWidth / w + ")";
    };
    frame.addEventListener("load", function () {
      // Sans dimensions connues d'avance, on les lit dans la page chargée.
      if (!frame.dataset.width) {
        try {
          var meta = frame.contentDocument.querySelector('meta[name="viewport"]');
          var m = /width=(\d+).*height=(\d+)/.exec(meta ? meta.content : "");
          if (m) { frame.dataset.width = m[1]; frame.dataset.height = m[2]; }
        } catch (e) { /* cadre inaccessible : on reste aux proportions par défaut */ }
      }
      fit();
    });
    if (window.ResizeObserver) new ResizeObserver(fit).observe(box); else window.addEventListener("resize", fit);
    fit();
  }

  // --- Réglages : sonder un service ---------------------------------------------------------
  function tester(button) {
    button.addEventListener("click", function () {
      var out = document.getElementById("verdict-" + button.dataset.test);
      button.disabled = true;
      if (out) out.innerHTML = '<span class="chip muted">…</span>';
      fetch("/settings/test/" + button.dataset.test, { method: "POST" })
        .then(function (r) { return r.text(); })
        .then(function (html) { if (out) out.innerHTML = html; })
        .catch(function () { if (out) out.innerHTML = '<span class="chip bad">injoignable</span>'; })
        .then(function () { button.disabled = false; });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-poll]").forEach(start);
    document.querySelectorAll("form[data-confirm]").forEach(guard);
    document.querySelectorAll("input[type=range][data-out]").forEach(bindRange);
    document.querySelectorAll("form[data-autosubmit]").forEach(autosubmit);
    document.querySelectorAll("input[type=file][data-submit]").forEach(submitOnPick);
    document.querySelectorAll(".inline-edit").forEach(inlineEdit);
    document.querySelectorAll(".listen").forEach(listen);
    // Le passage autour d'un segment signalé se déplie sous sa ligne.
    document.querySelectorAll(".context-toggle").forEach(function (button) {
      button.addEventListener("click", function () {
        var row = button.closest("tr").nextElementSibling;
        if (row && row.classList.contains("context")) row.hidden = !row.hidden;
      });
    });
    document.querySelectorAll("[data-test]").forEach(tester);
    var filter = document.getElementById("filter"); if (filter) libraryFilter(filter);
    var drop = document.getElementById("drop"); if (drop) dropzone(drop);
    var picker = document.getElementById("engine-picker"); if (picker) enginePicker(picker);
    var extra = document.getElementById("extra-backend"); if (extra) voiceList(extra, "voice-options");
    var direct = document.getElementById("direct-backend"); if (direct) voiceList(direct, "voice-options-direct");
    review();
  });
})();
