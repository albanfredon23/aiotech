// Mesure d'audience GoatCounter : statistiques anonymes, sans cookie, adresse IP non conservée.
// Rien n'est chargé si le visiteur l'a refusée (page Confidentialité), si son navigateur envoie
// Do Not Track ou Global Privacy Control, ou hors du site publié (copies locales, aperçus).
(() => {
  "use strict";
  const HOSTS = ["albanfredon23.github.io"];
  const KEY = "aiotech-stats";
  const read = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
  const write = (v) => { try { if (v) localStorage.setItem(KEY, v); else localStorage.removeItem(KEY); return true; } catch (e) { return false; } };
  const signal = navigator.doNotTrack === "1" || window.doNotTrack === "1" || navigator.globalPrivacyControl === true;

  if (read() !== "off" && !signal && HOSTS.includes(location.hostname)) {
    const s = document.createElement("script");
    s.async = true;
    s.src = "https://gc.zgo.at/count.js";
    s.dataset.goatcounter = "https://alban.goatcounter.com/count";
    document.head.appendChild(s);
  }

  // Page Confidentialité : bouton pour refuser ou réactiver la mesure d'audience.
  const btn = document.getElementById("stats-toggle"), out = document.getElementById("stats-state");
  if (!btn || !out) return;
  const render = () => {
    const off = read() === "off";
    out.textContent = signal
      ? "Votre navigateur demande à ne pas être suivi : la mesure d'audience est désactivée pour vous."
      : off ? "Mesure d'audience refusée sur ce navigateur." : "Mesure d'audience active (anonyme, sans cookie).";
    btn.textContent = off ? "Réactiver la mesure d'audience" : "Refuser la mesure d'audience";
    btn.hidden = signal;
  };
  btn.addEventListener("click", () => {
    if (!write(read() === "off" ? null : "off")) out.textContent = "Préférence impossible à enregistrer : le stockage local est bloqué par le navigateur.";
    else render();
  });
  render();
})();
