// Rafraîchissement de fragments, façon HTMX mais réduit à ce dont on se sert.
//
// Le seul besoin dynamique de l'interface est de suivre l'avancement d'une tâche : un
// fragment HTML relu périodiquement suffit, et cela évite d'embarquer une bibliothèque
// entière pour quinze lignes. Le rendu reste fait par le serveur.
//
//   <div data-poll="/chemin" data-interval="2000">…</div>
//
// Le sondage s'arrête de lui-même quand le fragment ne contient plus de tâche en cours,
// pour ne pas interroger le serveur indéfiniment sur un projet au repos.

(function () {
  "use strict";

  function refresh(node) {
    fetch(node.dataset.poll, { headers: { "X-Requested-With": "fetch" } })
      .then(function (response) {
        if (!response.ok) throw new Error(response.status);
        return response.text();
      })
      .then(function (html) {
        node.innerHTML = html;
        // Une tâche encore active porte la classe « running » : sans elle, plus rien
        // ne bouge et le sondage n'a plus lieu d'être.
        if (!node.querySelector(".job.running")) stop(node);
      })
      .catch(function () {
        stop(node);
      });
  }

  function stop(node) {
    if (node._timer) {
      clearInterval(node._timer);
      node._timer = null;
    }
  }

  function start(node) {
    var interval = parseInt(node.dataset.interval || "2000", 10);
    refresh(node);
    node._timer = setInterval(function () {
      refresh(node);
    }, interval);
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-poll]").forEach(start);
  });
})();
