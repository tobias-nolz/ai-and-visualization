// Chapter 5: switch off the forest. Outcome shares of the forest composites with the most tree-leaning units
// switched off vs. random units; where the former "empty" answers go; real test accuracy.
// Controls: tree- or ground-leaning units, how many (k, precomputed grid), and whether to show the random controls.
// Default (tree, k = 256, random shown): the numbers of concepts_ablation_*.csv in ab.runs.

import { OUTCOMES, OUTCOME_COLOR, OUTCOME_LABEL, INK, onResize, pct, tooltip } from "./format.js";
import { checkbox, slider, toggle } from "./controls.js";

const GREY_MARK = "#8a8984";

export function initAblation(ab) {
  const fig = d3.select("#ablation-figure");
  const body = fig.select(".chart-body");
  const tip = tooltip(body);
  const asIs = ab.runs["as is"];
  const defaultK = ab.runs["units off"].n_units;
  const state = { set: "tree", k: defaultK, random: true };

  const items = fig.select(".legend").selectAll(".item").data(OUTCOME_LABEL).join("span").attr("class", "item");
  items.append("i").attr("class", "sq").style("background", (_, i) => OUTCOME_COLOR[i]);
  items.append("span").text((d) => d);

  /** Runs for the current set and k: the stored ablation for the default, else the precomputed grid. */
  function runs() {
    if (state.set === "tree" && state.k === defaultK) {
      return { "as is": asIs, "units off": ab.runs["units off"], random: ab.runs.random,
        "random, matched": ab.runs["random, matched"], emptyCounts: ab.empty_after, emptyShare: ab.empty_after_share };
    }
    const d = ab.dose.sets[state.set][ab.dose.k.indexOf(state.k)];
    const counts = d.none.empty_after;
    return { "as is": asIs, "units off": d.none, random: d.random, "random, matched": d["random, matched"],
      emptyCounts: counts, emptyShare: Object.fromEntries(OUTCOMES.map((o) => [o, counts[o] / ab.empty_before])) };
  }

  const n = d3.format(",");
  const acc = [
    { label: "All test photos", key: "test|all" },
    { label: "Forest cameras", key: "test|forest" },
  ];

  // ------------------------------------------------------------------ controls
  const controls = fig.select(".controls");
  toggle(controls, {
    label: "Switch off",
    options: [{ value: "tree", label: "tree-leaning" }, { value: "ground", label: "ground-leaning" }],
    value: state.set,
    onChange: (v) => { state.set = v; draw(); },
  });
  slider(controls, {
    label: "units:",
    values: ab.dose.k,
    value: state.k,
    width: 110,
    onInput: (v) => { state.k = v; draw(); },
  });
  checkbox(controls, {
    label: "random controls",
    checked: state.random,
    onChange: (v) => { state.random = v; draw(); },
  });

  function stacked(g, y, h, shares, x0, x1, tipHtml) {
    let acc0 = 0;
    OUTCOMES.forEach((o, i) => {
      const v = shares[o];
      const xa = x0 + acc0 * (x1 - x0);
      const w = v * (x1 - x0);
      acc0 += v;
      g.append("rect").attr("x", xa + (i ? 1 : 0)).attr("width", Math.max(0, w - (i ? 1 : 0))).attr("y", y)
        .attr("height", h).attr("fill", OUTCOME_COLOR[i])
        .on("mousemove", (event) => tip.show(event, tipHtml(o, i)))
        .on("mouseleave", tip.hide);
      if (w > 34) {
        g.append("text").attr("class", `seg-label${o === "empty" ? " on-light" : ""}`)
          .attr("x", xa + w / 2).attr("y", y + h / 2).attr("dy", "0.35em").attr("text-anchor", "middle").text(pct(v));
      }
    });
  }

  function draw() {
    const W = body.node().getBoundingClientRect().width;
    if (W < 50) return;
    const r = runs();
    const { k } = state;
    const bars = [
      { label: "Model as is", run: r["as is"] },
      { label: `${k} ${state.set}-leaning units off`, run: r["units off"], strong: true },
      ...(state.random ? [
        { label: `${k} random units off`, run: r.random, sd: true },
        { label: "… matched on activation", run: r["random, matched"], sd: true },
      ] : []),
    ];
    const empty = {
      label: `Former “empty” answers (${n(ab.empty_before)})`,
      share: r.emptyShare,
      counts: r.emptyCounts,
    };
    const marks = [
      { key: "as is", label: "as is", shape: "ring" },
      { key: "units off", label: `${state.set} units off`, shape: "dot" },
      ...(state.random ? [{ key: "random", label: "random units off", shape: "square" }] : []),
    ];
    const labelW = Math.min(200, W * 0.32);
    const x0 = labelW;
    const x1 = W - 4;
    const barH = 22;
    const step = 32;
    const accTop = bars.length * step + 30 + step + 40;
    const H = accTop + acc.length * 26 + 34;
    const svg = body.select("svg").attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const g = svg.append("g");

    bars.forEach((b, i) => {
      const y = i * step;
      g.append("text").attr("class", "bar-label").attr("x", x0 - 10).attr("y", y + barH / 2).attr("dy", "0.35em")
        .attr("text-anchor", "end").style("font-weight", b.strong ? 650 : 400).text(b.label);
      stacked(g, y, barH, b.run.share, x0, x1, (o, j) => `<b>${b.label}</b><br>${OUTCOME_LABEL[j]}: ` +
        `<b>${pct(b.run.share[o])}</b>` + (b.sd ? ` (sd ${pct(b.run.sd[o])} over ${ab.draws} draws)` : ""));
    });

    // where the composites the model called "empty" go when the tree units are off
    const ye = bars.length * step + 30;
    g.append("text").attr("class", "tick").attr("x", x0).attr("y", ye - 8)
      .text(`With the ${k} ${state.set}-leaning units off, the answers that were “empty” become:`);
    g.append("text").attr("class", "bar-label").attr("x", x0 - 10).attr("y", ye + barH / 2).attr("dy", "0.35em")
      .attr("text-anchor", "end").text(empty.label);
    stacked(g, ye, barH, empty.share, x0, x1, (o, j) => `<b>${empty.label}</b><br>now ${OUTCOME_LABEL[j]}: ` +
      `<b>${n(empty.counts[o])}</b> (${pct(empty.share[o])})`);

    // real test accuracy: dot plot
    const xs = d3.scaleLinear()
      .domain([Math.floor(d3.min(acc, (a) => d3.min(marks, (mk) => r[mk.key].acc[a.key].acc)) * 50) / 50,
        Math.ceil(d3.max(acc, (a) => d3.max(marks, (mk) => r[mk.key].acc[a.key].acc)) * 50) / 50])
      .range([x0 + 8, x1 - 8]);
    g.append("text").attr("class", "grid-title").attr("x", 0).attr("y", accTop - 18).text("Real test photos: accuracy");
    const lg = g.append("g").attr("transform", `translate(${x0},${accTop - 22})`);
    let lx = 0;
    for (const mk of marks) {
      symbol(lg, lx + 5, 0, mk.shape);
      const t = lg.append("text").attr("class", "tick").attr("x", lx + 14).attr("y", 0).attr("dy", "0.35em")
        .text(mk.label);
      lx += 14 + t.node().getComputedTextLength() + 16;
    }
    acc.forEach((a, i) => {
      const y = accTop + i * 26 + 10;
      g.append("text").attr("class", "bar-label").attr("x", x0 - 10).attr("y", y).attr("dy", "0.35em")
        .attr("text-anchor", "end").text(a.label);
      g.append("line").attr("class", "gridline").attr("x1", x0).attr("x2", x1).attr("y1", y).attr("y2", y);
      const vals = marks.map((mk) => r[mk.key].acc[a.key].acc);
      g.append("line").attr("x1", xs(d3.min(vals))).attr("x2", xs(d3.max(vals))).attr("y1", y).attr("y2", y)
        .attr("stroke", "#d6d5cf").attr("stroke-width", 2);
      for (const mk of [...marks].reverse()) { // random first, so the "as is" ring stays visible on top
        const v = r[mk.key].acc[a.key];
        symbol(g, xs(v.acc), y, mk.shape)
          .on("mousemove", (event) => tip.show(event, `<b>${a.label}</b>, ${mk.label}<br>accuracy ` +
            `<b>${d3.format(".1%")(v.acc)}</b> on ${n(asIs.acc[a.key].n)} photos` +
            (mk.key === "random" ? ` (sd ${d3.format(".1%")(v.sd)})` : "")))
          .on("mouseleave", tip.hide);
      }
      const before = r["as is"].acc[a.key].acc;
      const off = r["units off"].acc[a.key].acc;
      g.append("text").attr("class", "bar-val").attr("x", x1).attr("y", y - 11).attr("text-anchor", "end")
        .text(`${d3.format(".1%")(before)} → ${d3.format(".1%")(off)}`);
    });
    for (const t of xs.ticks(4)) {
      g.append("text").attr("class", "tick").attr("x", xs(t)).attr("y", H - 6).attr("text-anchor", "middle")
        .text(d3.format(".0%")(t));
    }
  }

  function symbol(g, cx, cy, shape) {
    if (shape === "square") {
      return g.append("rect").attr("x", cx - 4.5).attr("y", cy - 4.5).attr("width", 9).attr("height", 9)
        .attr("fill", GREY_MARK);
    }
    return g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", shape === "ring" ? 5 : 5.5)
      .attr("fill", shape === "ring" ? "#fff" : INK).attr("stroke", INK).attr("stroke-width", 1.7);
  }

  onResize(draw);
}
