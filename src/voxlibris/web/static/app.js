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

  function isRunning(node) {
    return !!node.querySelector(".job.running");
  }

  function refresh(node) {
    var wasRunning = isRunning(node);
    fetch(node.dataset.poll, { headers: { "X-Requested-With": "fetch" } })
      .then(function (response) {
        if (!response.ok) throw new Error(response.status);
        return response.text();
      })
      .then(function (html) {
        node.innerHTML = html;
        if (!isRunning(node)) {
          stop(node);
          // Le fragment n'est pas seul concerné par la fin d'une tâche : les étapes
          // suivantes viennent de se débloquer, des compteurs ont changé, un lien de
          // téléchargement est apparu. Tout cela a été rendu par le serveur avant que
          // la tâche ne s'achève, et se trouve donc périmé. Sans ce rechargement,
          // l'utilisateur reste devant des boutons grisés sans savoir pourquoi.
          if (wasRunning) window.location.reload();
        }
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
