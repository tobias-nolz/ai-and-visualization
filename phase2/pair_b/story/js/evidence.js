// Chapter 2: where is the evidence? A wrong composite with its two class maps on the 14 x 14 grid, and the winner's
// lead split into the animal's cells and the background, for that composite and averaged over all wrong composites
// per habitat. Default: the forest hero composite, each cell coloured by the class whose map is higher there.
// Controls: the animal and habitat of the composite (one wrong composite each, see evidence.pick_rule), and which
// map to show (difference, the true species' map, the winning class's map).

import { HABITAT_COLOR, TRUE_COLOR, WINNER_COLOR, logit, onResize, pct, tooltip } from "./format.js";
import { dropdown, toggle } from "./controls.js";

const HABITAT_ROWS = ["forest", "mixed", "open"];
const MAX_ALPHA = 0.78;
const aFrame = (h) => `${h === "open" ? "an" : "a"} ${h} frame`;
const DEFAULT = { group: "elephant", habitat: "forest", mode: "diff" };
const MODES = [
  { value: "diff", label: "difference" },
  { value: "true", label: "true species" },
  { value: "pred", label: "winning class" },
];

export function initEvidence(ev) {
  const fig = d3.select("#evidence-figure");
  const tip = tooltip(fig.select(".chart-body"));
  const picks = new Map(ev.picks.map((p) => [`${p.group}|${p.habitat}`, p.rule === "hero composite" ? { ...p, ...ev.hero } : p]));
  const groups = [...new Set(ev.picks.map((p) => p.group))];
  let state = { ...DEFAULT };
  const pick = () => picks.get(`${state.group}|${state.habitat}`);
  const isHero = (p) => p.rule === "hero composite";

  // ------------------------------------------------------------------ head: subtitle, legend, controls
  const sub = fig.select(".fig-sub");
  const defaultSub = sub.text();
  const legend = fig.select(".legend");
  const defaultLegend = legend.html();

  function setHead(p) {
    const what = isHero(p) ? "the forest hero composite" : `the ${p.true} cut-out in ${aFrame(p.habitat)}`;
    const how = {
      diff: "each grid cell coloured by the class whose map is higher there",
      true: `each grid cell by the ${p.true} map (brighter = higher)`,
      pred: `each grid cell by the ${p.pred} map (brighter = higher, same scale)`,
    }[state.mode];
    sub.text(isHero(p) && state.mode === "diff" ? defaultSub
      : `Left: ${what}, ${how}. Right: the same split, averaged over all wrong composites.`);
    if (state.mode === "diff") legend.html(defaultLegend);
    else {
      const color = state.mode === "true" ? TRUE_COLOR : WINNER_COLOR;
      legend.html(`<span class="item"><i class="sq" style="background:${color}"></i>${state.mode === "true"
        ? `${p.true} (true species)` : `${p.pred} (winning class)`}: higher score</span>` +
        `<span class="item"><i class="ring sq"></i>the animal's cells</span>`);
    }
  }

  const controls = fig.select(".controls");
  const animal = dropdown(controls, {
    label: "Animal",
    options: groups.map((g) => ({ value: g, label: g })),
    value: state.group,
    onChange: (g) => {
      state.group = g;
      if (!pick()) state.habitat = HABITAT_ROWS.find((h) => picks.has(`${g}|${h}`));
      update();
    },
  });
  const habitat = toggle(controls, {
    label: "Habitat",
    options: HABITAT_ROWS.map((h) => ({ value: h, label: h })),
    value: state.habitat,
    onChange: (h) => { state.habitat = h; update(); },
  });
  const mode = toggle(controls, {
    label: "Map",
    options: MODES,
    value: state.mode,
    onChange: (m) => { state.mode = m; update(); },
  });

  // ------------------------------------------------------------------ the composite and its class maps
  const img = fig.select(".ev-stage img");
  const overlay = fig.select(".ev-overlay").attr("preserveAspectRatio", "none");

  function showPhoto(p) {
    img.attr("src", `img/${p.img}`).attr("width", p.width).attr("height", p.height)
      .attr("alt", `The ${p.true} cut-out in ${aFrame(p.habitat || "forest")} (camera ${p.camera}); the model says ` +
        `${p.pred}, ${pct(p.p_pred)}. Grid cells are tinted blue where the ${p.true} map is higher, orange where ` +
        `the ${p.pred} map is higher.`);
    const n = p.grid;
    overlay.attr("viewBox", `0 0 ${n} ${n}`);
    let cells;
    if (state.mode === "diff") {
      const diff = p.cam_pred.map((row, i) => row.map((v, j) => v - p.cam_true[i][j])); // > 0: favours the winner
      const maxAbs = d3.max(diff.flat(), Math.abs);
      cells = diff.flatMap((row, i) => row.map((v, j) => ({
        i, j, fill: v > 0 ? WINNER_COLOR : TRUE_COLOR, alpha: MAX_ALPHA * Math.abs(v) / maxAbs,
      })));
    } else {
      // one map; the scale spans both maps, so the two views are comparable
      const all = [...p.cam_true.flat(), ...p.cam_pred.flat()];
      const [lo, hi] = d3.extent(all);
      const cam = state.mode === "true" ? p.cam_true : p.cam_pred;
      const fill = state.mode === "true" ? TRUE_COLOR : WINNER_COLOR;
      cells = cam.flatMap((row, i) => row.map((v, j) => ({ i, j, fill, alpha: MAX_ALPHA * (v - lo) / (hi - lo) })));
    }
    overlay.selectAll("rect.cell").data(cells).join("rect").attr("class", "cell")
      .attr("x", (d) => d.j).attr("y", (d) => d.i).attr("width", 1).attr("height", 1)
      .attr("fill", (d) => d.fill)
      .attr("fill-opacity", (d) => d.alpha);
    // outline of the animal's cells (cells at least half covered by the cut-out)
    const inside = (i, j) => i >= 0 && j >= 0 && i < n && j < n && p.mask[i][j] >= 0.5;
    const edges = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!inside(i, j)) continue;
        if (!inside(i - 1, j)) edges.push([j, i, j + 1, i]);
        if (!inside(i + 1, j)) edges.push([j, i + 1, j + 1, i + 1]);
        if (!inside(i, j - 1)) edges.push([j, i, j, i + 1]);
        if (!inside(i, j + 1)) edges.push([j + 1, i, j + 1, i + 1]);
      }
    }
    overlay.selectAll("line.edge").data(edges).join("line").attr("class", "edge")
      .attr("x1", (d) => d[0]).attr("y1", (d) => d[1]).attr("x2", (d) => d[2]).attr("y2", (d) => d[3])
      .attr("stroke", "#fff").attr("stroke-width", 2.2).attr("vector-effect", "non-scaling-stroke");
    const where = isHero(p) ? "a forest frame" : `${aFrame(p.habitat)}, pasted at the ${p.cell}`;
    const rule = isHero(p) ? "" : " The model's most confident wrong answer for this animal here.";
    // the picks are the most confident wrong answers, so the animal often favours the winner too; set it against
    // the average over all wrong composites in this habitat
    let atypical = "";
    if (p.animal > 0) {
      const avg = ev.habitats[p.habitat || "forest"];
      const lean = Math.abs(avg.animal) < 0.1 ? "are about neutral"
        : avg.animal < 0 ? "lean to the true species" : "lean to the winning class";
      atypical = ` Here even the animal's cells favour the ${p.pred}. Over all ${d3.format(",")(avg.n)} wrong ` +
        `${p.habitat || "forest"} composites they ${lean} (${d3.format("+.2f")(avg.animal)}).`;
    }
    fig.select(".ev-caption").html(
      `<b>${p.true}</b> in ${where}: the model says <b>${p.pred}</b> (${pct(p.p_pred)}). ` +
      `Its lead: ${logit(p.animal)} on the animal's cells, ${logit(p.background)} on the background.${rule}` +
      atypical);
  }

  // ------------------------------------------------------------------ bars: lead from animal cells vs. background
  const averages = HABITAT_ROWS.map((k) => ({
    key: k, label: k, sub: `${d3.format(",")(ev.habitats[k].n)} wrong`, habitat: true, ...ev.habitats[k],
  }));

  function draw() {
    const rows = [{ key: "hero", label: "This composite", sub: "", ...pick() }, ...averages];
    const svg = fig.select("svg.ev-bars");
    const photoH = fig.select(".ev-stage").node().getBoundingClientRect().height;
    const W = fig.select(".chart-body").node().getBoundingClientRect().width;
    if (W < 50) return;
    const H = Math.max(230, photoH);
    svg.attr("viewBox", `0 0 ${W} ${H}`).selectAll("*").remove();
    const m = { top: 18, right: 46, bottom: 40, left: 92 };
    const lo = Math.min(-2, d3.min(rows, (d) => d.animal));
    const hi = d3.max(rows, (d) => d.background);
    const x = d3.scaleLinear().domain([lo, hi]).range([m.left, W - m.right]).nice();
    const band = (H - m.top - m.bottom) / rows.length;
    const barH = Math.min(14, band / 3.2);
    const g = svg.append("g");

    for (const t of x.ticks(5)) {
      g.append("line").attr("class", "gridline").attr("x1", x(t)).attr("x2", x(t))
        .attr("y1", m.top - 6).attr("y2", H - m.bottom + 4);
      g.append("text").attr("class", "tick").attr("x", x(t)).attr("y", H - m.bottom + 18)
        .attr("text-anchor", "middle").text(t === 0 ? "0" : logit(t));
    }
    g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.top - 6).attr("y2", H - m.bottom + 4)
      .attr("stroke", "#8a8984");
    g.append("text").attr("class", "axis-title").attr("x", x(0) - 6).attr("y", H - 4).attr("text-anchor", "end")
      .text("← true species");
    g.append("text").attr("class", "axis-title").attr("x", x(0) + 6).attr("y", H - 4).text("winning class →");
    // divider between the example and the averages
    g.append("line").attr("x1", 0).attr("x2", W).attr("y1", m.top + band - 2).attr("y2", m.top + band - 2)
      .attr("stroke", "#e7e6e1");
    g.append("text").attr("class", "tick").attr("x", 0).attr("y", m.top + band + 12)
      .text("All wrong composites:");

    rows.forEach((d, k) => {
      const y0 = m.top + k * band + (k ? band * 0.55 : band * 0.35);
      const row = g.append("g").attr("transform", `translate(0,${y0})`);
      row.append("text").attr("class", "bar-label").attr("x", m.left - 10).attr("y", 0).attr("dy", "0.35em")
        .attr("text-anchor", "end").style("font-weight", d.habitat ? 600 : 650)
        .style("fill", d.habitat ? HABITAT_COLOR[d.key] : null).text(d.label);
      if (d.sub) {
        row.append("text").attr("class", "tick").attr("x", m.left - 10).attr("y", 13).attr("text-anchor", "end")
          .text(d.sub);
      }
      for (const [part, v, dy] of [["animal", d.animal, -barH / 2 - 1], ["background", d.background, barH / 2 + 1]]) {
        const x0 = x(Math.min(0, v));
        const w = Math.max(1, Math.abs(x(v) - x(0)));
        row.append("rect").attr("x", x0).attr("width", w).attr("y", dy - barH / 2).attr("height", barH)
          .attr("rx", 2).attr("fill", v > 0 ? WINNER_COLOR : TRUE_COLOR)
          .on("mousemove", (event) => tip.show(event, `<b>${d.key === "hero" ? "This composite" : d.label}</b>` +
            `<br>lead from the ${part === "animal" ? "animal's cells" : "background"}: <b>${logit(v)}</b>` +
            (d.key === "hero" ? "" : `<br>${d3.format(",")(d.n)} wrong composites; the animal covers ` +
              `${pct(d.mask_area)} of the frame`)))
          .on("mouseleave", tip.hide);
        // value labels right of the bar end, or right of zero for the (short, negative) animal bars
        row.append("text").attr("class", "bar-val").attr("y", dy).attr("dy", "0.35em")
          .attr("x", x(Math.max(0, v)) + 5)
          .text(`${logit(v)} ${part === "animal" ? "animal" : "background"}`);
      }
    });
  }

  function update() {
    const p = pick();
    animal.set(state.group);
    habitat.set(state.habitat);
    habitat.enable((h) => picks.has(`${state.group}|${h}`));
    mode.set(state.mode);
    setHead(p);
    showPhoto(p);
    // the bars take the photo's height: redraw once the new picture has its size
    if (img.node().complete) draw();
    else img.node().addEventListener("load", draw, { once: true });
  }

  update();
  // first draw (and redraw on resize) once the image has its size
  const node = img.node();
  if (node.complete) onResize(draw);
  else node.addEventListener("load", () => onResize(draw), { once: true });
}
