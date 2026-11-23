// Chapter 3: what does the model see in the trees? Every layer4 unit of the fine-tuned model as one square,
// labelled by the paper's rule (IoU of its strongest activations) and by where its total activation goes; plus
// three example units with their frames. Two sliders change the thresholds of the two rules; the mosaics recolour
// (each unit keeps its square, sorted by the default labels).

import { CONCEPT_COLOR, onResize, pct, tooltip } from "./format.js";
import { slider } from "./controls.js";
import { introAnimation, tr } from "./reveal.js";

const C = CONCEPT_COLOR.dark;
const ORDER = ["animal", "tree", "ground", "sky", "plant", "grass", "rock", "water", "info strip", "other", "mixed",
  "dead"];
const COLS = 64;
const OTHER = "#6f6e69"; // concepts without their own colour (only with lowered thresholds)
const color = (label) => C[label] || (label === "" ? C.none : OTHER);
const RING_MAX = 24; // ring the units labelled by the paper's rule only while they are few
const SUB_MAX = 4; // concepts listed under a mosaic's title

export function initUnits(u) {
  const fig = d3.select("#units-figure");
  const body = fig.select(".chart-body");
  const tip = tooltip(body);
  const units = u.units.map(([act, iou, excess, share, best, bestIou, bestAct, lift], unit) => ({
    unit, act: u.labels[act], iou: u.iou_labels[iou], excess, share, best: u.iou_labels[best], bestIou,
    bestAct: u.labels[bestAct], lift,
  }));
  // same position in both mosaics: grouped by the activation label (default thresholds), strongest first
  units.sort((a, b) => ORDER.indexOf(a.act) - ORDER.indexOf(b.act) || b.excess - a.excess);
  const examples = new Set(u.examples.map((e) => e.unit));

  // the two rules with adjustable thresholds; with the defaults they give the stored labels
  const t = { iou: u.rules.iou_min, excess: u.rules.min_excess };
  const relabel = () => units.forEach((d) => {
    d.iou = d.bestIou >= t.iou ? d.best : "";
    d.act = d.excess >= t.excess && d.lift >= u.rules.min_lift ? d.bestAct : "mixed";
  });
  relabel();

  const fmt = d3.format(",");
  const defaultShown = ORDER.filter((l) => u.counts.finetuned[l]);
  function setLegend() {
    // the default legend, plus colours that only appear with other thresholds
    const used = new Set(units.flatMap((d) => [d.act, d.iou]));
    const named = [...defaultShown.filter((l) => l !== "mixed"), "water"];
    const extra = ORDER.filter((l) => used.has(l) && C[l] && !named.includes(l) && l !== "mixed");
    const legend = [...named, ...extra].map((l) => ({ label: l, color: C[l] }));
    if ([...used].some((l) => l && !C[l] && l !== "mixed")) legend.push({ label: "other concepts", color: OTHER });
    legend.push({ label: "mixed / no label", color: C.mixed });
    const items = fig.select(".legend").selectAll(".item").data(legend).join("span").attr("class", "item").html("");
    items.append("i").attr("class", "sq").style("background", (d) => d.color);
    items.append("span").text((d) => d.label);
  }

  function panels() {
    const paperCounts = d3.rollup(units, (v) => v.length, (d) => d.iou || "none");
    const actCounts = d3.rollup(units, (v) => v.length, (d) => d.act);
    const labelled = units.length - (paperCounts.get("none") || 0);
    // at most SUB_MAX concepts per subtitle (more only appear with low thresholds), the rest summed up
    const list = (entries, sep) => {
      const rest = d3.sum(entries.slice(SUB_MAX), (e) => e[1]);
      return entries.slice(0, SUB_MAX).map(([k, v]) => `${fmt(v)} ${k}`)
        .concat(rest ? [`${fmt(rest)} other`] : []).join(sep);
    };
    const paper = [...paperCounts].filter(([k]) => k !== "none").sort((a, b) => b[1] - a[1]);
    const act = ORDER.filter((l) => actCounts.get(l) && l !== "mixed").map((l) => [l, actCounts.get(l)]);
    return [
      { key: "iou", title: "Paper's rule: where it fires hardest",
        sub: `${fmt(labelled)} labelled` + (labelled ? `: ${list(paper, ", ")}` : "") },
      { key: "act", title: "Where all its activation goes",
        sub: `${list(act, " · ")} · ${fmt(actCounts.get("mixed") || 0)} mixed` },
    ];
  }

  function unitTip(d) {
    const act = d.act === "mixed"
      ? `mixed (strongest: ${pct(d.share)} of its activation on ${d.bestAct})`
      : `<b>${d.act}</b>: ${pct(d.share)} of its activation there (area ${pct(u.area[d.act])})`;
    const iou = d.iou ? `<b>${d.iou}</b> (IoU ${d.bestIou})` : `not labelled (best: ${d.best}, IoU ${d.bestIou})`;
    return `<b>unit ${d.unit}</b><br>by total activation: ${act}<br>paper's rule: ${iou}`;
  }

  function draw() {
    const W = body.node().getBoundingClientRect().width;
    if (W < 50) return;
    const narrow = W < 560;
    const gap = 28;
    const s = Math.max(3, Math.floor((narrow ? W : (W - gap) / 2) / COLS));
    const mw = s * COLS;
    const mh = s * Math.ceil(units.length / COLS);
    const head = 38;
    const H = narrow ? 2 * (head + mh) + gap : head + mh;
    const svg = body.select("svg").attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    // intro (as in the slide clip): the few units the paper's rule labels light up and get ringed, then the
    // activation labels sweep through the second mosaic, group by group
    const mode = anim.mode;
    const labelled = units.filter((d) => d.act !== "mixed");
    const rank = new Map(labelled.map((d, i) => [d.unit, i]));
    const SWEEP = 2400;
    panels().forEach((p, k) => {
      const gx = narrow ? 0 : k * (mw + gap);
      const gy = narrow ? k * (head + mh + gap) : 0;
      const g = svg.append("g").attr("transform", `translate(${gx},${gy})`);
      g.append("text").attr("class", "grid-title").attr("y", 12).text(p.title);
      const sub = g.append("text").attr("class", "tick").attr("y", 28).text(p.sub);
      if (mode !== "end") sub.attr("opacity", 0);
      if (mode === "play") tr(sub, k ? 1400 + SWEEP : 500, 500).attr("opacity", 1);
      const cells = g.append("g").attr("transform", `translate(0,${head})`);
      const blank = p.key === "iou" ? C.none : C.mixed;
      const rects = cells.selectAll("rect").data(units).join("rect")
        .attr("x", (_, i) => (i % COLS) * s).attr("y", (_, i) => Math.floor(i / COLS) * s)
        .attr("width", s - 1).attr("height", s - 1)
        .attr("fill", (d) => (mode === "end" ? color(d[p.key]) : blank))
        .on("mousemove", (event, d) => tip.show(event, unitTip(d)))
        .on("mouseleave", tip.hide);
      if (mode === "play") {
        const delay = (d) => (p.key === "iou" ? 300 : 1300 + (rank.get(d.unit) / labelled.length) * SWEEP);
        tr(rects.filter((d) => color(d[p.key]) !== blank), delay, 300).attr("fill", (d) => color(d[p.key]));
      }
      // the few units labelled by the paper's rule are single squares: ring them
      const marked = units.map((d, i) => ({ d, i }))
        .filter(({ d }) => (p.key === "iou" ? d.iou !== "" : examples.has(d.unit)));
      if (p.key === "iou" && marked.length > RING_MAX) marked.length = 0;
      cells.selectAll("circle").data(marked).join("circle")
        .attr("cx", ({ i }) => (i % COLS) * s + (s - 1) / 2).attr("cy", ({ i }) => Math.floor(i / COLS) * s + (s - 1) / 2)
        .attr("r", s * 1.6).attr("fill", "none").attr("stroke", "#f3f2ee").attr("stroke-width", 1.4)
        .style("pointer-events", "none")
        .call((rings) => {
          if (mode !== "end") rings.attr("opacity", 0);
          if (mode !== "play") return;
          if (p.key === "iou") {
            rings.attr("r", s * 9);
            tr(rings, 300, 800, d3.easeBackOut).attr("r", s * 1.6).attr("opacity", 1);
          } else tr(rings, 1400 + SWEEP, 500).attr("opacity", 1);
        });
      if (p.key === "act") {
        cells.selectAll("text.ex").data(marked).join("text").attr("class", "tick ex")
          .attr("x", ({ i }) => (i % COLS) * s + s * 2.3 + 2).attr("y", ({ i }) => Math.floor(i / COLS) * s + s / 2)
          .attr("dy", "0.35em").attr("stroke", "#151514").attr("stroke-width", 3).attr("paint-order", "stroke")
          .style("fill", "#f3f2ee").style("font-weight", 600).style("pointer-events", "none")
          .text(({ d }) => d.unit)
          .call((lbl) => {
            if (mode !== "end") lbl.attr("opacity", 0);
            if (mode === "play") tr(lbl, 1400 + SWEEP, 500).attr("opacity", 1);
          });
      }
    });
  }
  const anim = introAnimation(fig, draw);
  setLegend();
  onResize(draw);

  // ------------------------------------------------------------------ threshold sliders
  const controls = fig.select(".controls");
  const update = () => { anim.skip(); relabel(); setLegend(); draw(); };
  slider(controls, {
    label: "Paper's rule: IoU at least",
    values: d3.range(5, 51).map((v) => v / 1000),
    value: t.iou,
    format: (v) => v.toFixed(3),
    onInput: (v) => { t.iou = v; update(); },
  }).node.attr("title", "A unit gets a concept if its strongest activations overlap the concept's pixels at least " +
    "this much (IoU); the paper uses 0.04.");
  slider(controls, {
    label: "Total activation: excess at least",
    values: d3.range(0, 31).map((v) => v / 100),
    value: t.excess,
    format: (v) => `${Math.round(v * 100)} pts`,
    onInput: (v) => { t.excess = v; update(); },
  }).node.attr("title", "A unit gets a concept if that concept holds at least this many percentage points more of " +
    `its activation than of the frame (and ${u.rules.min_lift}× its area); we use 10.`);

  // ------------------------------------------------------------------ example units
  const cards = fig.select(".unit-examples").selectAll(".unit-card").data(u.examples).join("div")
    .attr("class", "unit-card");
  const h4 = cards.append("h4");
  h4.append("i").style("background", (d) => C[d.label]);
  h4.append("span").text((d) => `Unit ${d.unit} · ${d.label}`);
  cards.append("p").text((d) =>
    `${pct(d.share)} of its activation on ${d.label === "animal" ? "the animal" : d.label}, which covers ` +
    `${pct(d.area)} of the frames. Paper's rule: ${d.iou >= u.rules.iou_min ? d.iou_concept : "no label"} ` +
    `(IoU ${d.iou}).`);
  cards.append("div").attr("class", "thumbs").selectAll("img").data((d) => d.frames.map((f) => ({ ...f, unit: d })))
    .join("img")
    .attr("src", (f) => `img/${f.img}`).attr("width", (f) => f.width).attr("height", (f) => f.height)
    .attr("loading", "lazy")
    .attr("alt", (f) => `Frame from camera ${f.camera} (${f.habitat}); brighter where unit ${f.unit.unit} is more ` +
      `active.`);
}
