// Presenting: arrow keys, space and page up / down jump between the main-line steps ([data-beat]) with a smooth
// scroll. Content in closed "more" boxes is not a beat, so it is skipped.

const NEXT = new Set(["ArrowDown", "ArrowRight", "PageDown", " "]);
const PREV = new Set(["ArrowUp", "ArrowLeft", "PageUp"]);
const SETTLE_MS = 900; // a key pressed while the previous jump is still scrolling continues from its target

/** Scroll position at which a beat is "on screen": centred, or its top at the top if it is about a screen tall. */
function targetY(el) {
  const r = el.getBoundingClientRect();
  const vh = window.innerHeight;
  const top = r.top + window.scrollY;
  if (r.height >= 0.9 * vh) return Math.round(top);
  return Math.max(0, Math.round(top - (vh - r.height) / 2));
}

function typing(el) {
  return el.closest("input, textarea, select, [contenteditable]") !== null;
}

export function setupKeys(selector = "[data-beat]") {
  const beats = Array.from(document.querySelectorAll(selector));
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  let last = { index: -1, time: 0 };

  function go(dir) {
    const ys = beats.map(targetY);
    let i;
    if (performance.now() - last.time < SETTLE_MS && last.index >= 0) {
      i = last.index + dir;
    } else {
      const y = window.scrollY;
      i = dir > 0 ? ys.findIndex((t) => t > y + 4) : ys.findLastIndex((t) => t < y - 4);
    }
    if (i < 0 || i >= beats.length) return;
    last = { index: i, time: performance.now() };
    window.scrollTo({ top: ys[i], behavior: reduce.matches ? "auto" : "smooth" });
  }

  window.addEventListener("keydown", (event) => {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || typing(event.target)) return;
    // space on a focused box title, link or button keeps its normal meaning (open the box, follow the link)
    if (event.key === " " && event.target.closest("summary, a, button")) return;
    let dir = 0;
    if (event.key === " ") dir = event.shiftKey ? -1 : 1;
    else if (NEXT.has(event.key)) dir = 1;
    else if (PREV.has(event.key)) dir = -1;
    if (!dir) return;
    event.preventDefault();
    go(dir);
  });
  return { go };
}

/** Links to a closed box (or to something inside one) open it first, e.g. "#methods" in the contents. */
export function openBoxesForLinks() {
  const open = (hash) => {
    if (!hash || hash === "#") return;
    const el = document.getElementById(decodeURIComponent(hash.slice(1)));
    if (!el) return;
    for (let d = el.closest("details"); d; d = d.parentElement.closest("details")) d.open = true;
  };
  document.addEventListener("click", (event) => {
    const a = event.target.closest('a[href^="#"]');
    if (a) open(a.getAttribute("href"));
  });
  window.addEventListener("hashchange", () => open(location.hash));
  open(location.hash);
}
