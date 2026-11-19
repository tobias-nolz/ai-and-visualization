// Shared constants, number formats and the data binding for numbers in the text.

export const OUTCOMES = ["exact species", "same group", "other group", "empty"];
export const OUTCOME_COLOR = ["#2a78d6", "#1baf7a", "#eb6834", "#c9c8c2"];
export const OUTCOME_LABEL = ["correct (exact species)", "right group, other species", "other group", "predicted empty"];
export const HABITATS = ["forest", "mixed", "open"]; // left to right on the openness axis
export const HABITAT_COLOR = { open: "#eda100", mixed: "#8c9440", forest: "#008300" };
export const INK = "#1f1f1e";
export const MUTED = "#8a8984";
export const TRUE_COLOR = OUTCOME_COLOR[0]; // ch. 2: evidence for the true species
export const WINNER_COLOR = OUTCOME_COLOR[2]; // ch. 2: evidence for the winning (wrong) class

// concepts (ch. 3, 4); checked with the dataviz palette validator (CVD-safe, light and dark surface)
export const CONCEPT_COLOR = {
  light: { animal: "#8a5cc7", tree: "#1f7a3a", ground: "#d39a2c", sky: "#3f97d4", "info strip": "#52514e",
    "other scenery": "#b9b8b2" },
  dark: { animal: "#9470d6", tree: "#1f7a3a", ground: "#c08519", sky: "#3b92cf", water: "#2bb3a8",
    "info strip": "#8a8984", mixed: "#3a3a37", none: "#2c2c29" },
};

/** Value at a dotted path, e.g. get(data, "recap.cells.swap.rows.0"); keys may contain "|" or spaces. */
export function get(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

/** Whole percent; never shows 0 % or 100 % for values that are not exactly 0 or 1. */
export function pct(x) {
  const r = Math.round(x * 100);
  if (x > 0 && r === 0) return "<1%";
  if (x < 1 && r === 100) return ">99%";
  return `${r}%`;
}

/** Logits with a sign and one decimal, e.g. "+7.5", "−0.8" (true minus sign). */
export const logit = (x) => d3.format("+.1f")(x);

export const FORMATS = {
  pct,
  pct1: (x) => d3.format(".1%")(x),
  logit,
  int: (x) => d3.format(",")(x),
  text: (x) => String(x),
};

/** Fill every element with data-v="file.path" (and optional data-f="pct|int|text") from the loaded JSON. */
export function fillNumbers(root, data) {
  const missing = [];
  root.querySelectorAll("[data-v]").forEach((el) => {
    const v = get(data, el.dataset.v);
    if (v === undefined || v === null) {
      el.textContent = "??";
      el.classList.add("is-missing");
      missing.push(el.dataset.v);
      return;
    }
    el.textContent = (FORMATS[el.dataset.f] || FORMATS.text)(v);
  });
  if (missing.length) console.warn("numbers missing in data/*.json:", missing);
  return missing;
}

/** Shared hover tooltip inside a positioned container (a .tooltip element in it). */
export function tooltip(container) {
  const tip = container.select(".tooltip");
  return {
    show(event, html) {
      const box = container.node().getBoundingClientRect();
      const x = event.clientX - box.left;
      const y = event.clientY - box.top;
      tip.html(html).classed("is-on", true);
      const t = tip.node().getBoundingClientRect();
      const left = x + 14 + t.width > box.width ? x - t.width - 14 : x + 14;
      tip.style("left", `${Math.max(0, left)}px`).style("top", `${Math.max(0, y - t.height - 10)}px`);
    },
    hide() { tip.classed("is-on", false); },
  };
}

/** Call draw() now and again (debounced) when the window is resized. */
export function onResize(draw) {
  draw();
  let timer;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(draw, 150);
  });
}
