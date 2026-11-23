// Chapter 4: who votes for the great argus? Per habitat, every concept's share of the winner's lead (its "votes")
// against its share of the frame, over the wrong composites. On the diagonal a concept votes in proportion to its
// area. A toggle picks the winning class: all wrong answers (default) or one frequent winner, e.g. the great argus.

import { CONCEPT_COLOR, HABITAT_COLOR, onResize, pct, tooltip } from "./format.js";
import { toggle } from "./controls.js";
import { introAnimation, tr } from "./reveal.js";

const C = CONCEPT_COLOR.light;
const PANELS = ["forest", "mixed", "open"];
const CONCEPTS = ["tree", "ground", "sky", "animal", "info strip", "other scenery"];

export function initVotes(votes) {
  const fig = d3.select("#votes-figure");
  const body = fig.select(".chart-body");
  const tip = tooltip(body);

  const items = fig.select(".legend").selectAll(".item").data(CONCEPTS).join("span").attr("class", "item");
  items.append("i").style("background", (d) => C[d]);
  items.append("span").text((d) => d);

  let winner = "all";
  let morph = false; // after a change of the winning class, points glide from their old to their new position
  const prev = new Map(); // `${habitat}|${concept}` -> last drawn { area, share }
  const panel = (h) => (winner === "all" ? votes.habitats[h] : votes.by_winner[winner][h]);
  const sub = fig.select(".fig-sub");
  const defaultSub = sub.text();
  const fmt = d3.format(",");
  toggle(fig.select(".controls"), {
    label: "Winning class",
    options: [{ value: "all", label: "all wrong answers" },
      ...votes.winners.map((w) => ({ value: w.winner, label: w.winner,
        title: `${fmt(w.n)} wrong composites with this answer` }))],
    value: winner,
    onChange: (w) => {
      anim.skip();
      winner = w;
      morph = true;
      sub.text(w === "all" ? defaultSub : `Wrong composites the model calls ${w === "empty" ? "“empty”" : w}. ` +
        "On the dashed line, a concept votes exactly in proportion to its area.");
      draw();
    },
  });

  function draw() {
    // axes: the default range, widened if the picked winning class needs more
    const all = PANELS.flatMap((h) => [...votes.habitats[h].folded, ...(panel(h)?.folded || [])]);
    const xMax = Math.ceil(d3.max(all, (d) => d.area) * 10) / 10;
    const yMax = Math.ceil(d3.max(all, (d) => d.share) * 10) / 10;
    const yMin = Math.floor(d3.min(all, (d) => d.share) * 10) / 10;
    const top = Math.max(xMax, yMax);

    const W = body.node().getBoundingClientRect().width;
    if (W < 50) return;
    const narrow = W < 560;
    const gap = 22;
    const m = { top: 26, right: 10, bottom: 40, left: 50 };
    const pw = narrow ? W : (W - 2 * gap) / 3;
    const plotW = pw - m.left - m.right;
    const plotH = Math.min(plotW * (top - yMin) / top, 300);
    const ph = m.top + plotH + m.bottom;
    const H = narrow ? 3 * ph : ph;
    const svg = body.select("svg").attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const x = d3.scaleLinear().domain([0, top]).range([0, plotW]);
    const y = d3.scaleLinear().domain([yMin, top]).range([plotH, 0]);

    // intro (as in the slide clip): forest first, its diagonal drawn and its points flying out from the origin with the
    // trees ringed; then mixed and open
    const mode = anim.mode;
    const glide = morph && mode === "end";
    morph = false;
    PANELS.forEach((hab, k) => {
      const d = panel(hab);
      const t0 = k === 0 ? 200 : 2900 + (k - 1) * 800;
      const g = svg.append("g").attr("transform", narrow
        ? `translate(${m.left},${k * ph + m.top})` : `translate(${k * (pw + gap) + m.left},${m.top})`);
      if (k > 0 && mode !== "end") g.attr("opacity", 0);
      if (k > 0 && mode === "play") tr(g, t0 - 400, 500).attr("opacity", 1);
      g.append("text").attr("class", "panel-title").attr("y", -12).style("fill", HABITAT_COLOR[hab])
        .text(hab);
      if (d) {
        g.append("text").attr("class", "tick").attr("x", plotW).attr("y", -12).attr("text-anchor", "end")
          .text(`lead ${d3.format(".1f")(d.lead)} logits` + (winner === "all" ? "" : ` · ${fmt(d.n)}`));
      }
      for (const t of d3.range(0, top + 1e-9, 0.2)) {
        g.append("line").attr("class", "gridline").attr("x1", x(t)).attr("x2", x(t)).attr("y1", 0).attr("y2", plotH);
        g.append("text").attr("class", "tick").attr("x", x(t)).attr("y", plotH + 15).attr("text-anchor", "middle")
          .text(pct(t));
      }
      for (const t of d3.range(Math.ceil(yMin / 0.2) * 0.2, top + 1e-9, 0.2)) {
        const tt = Math.round(t * 10) / 10;
        g.append("line").attr("class", "gridline").attr("x1", 0).attr("x2", plotW).attr("y1", y(tt)).attr("y2", y(tt));
        if (k === 0 || narrow) {
          g.append("text").attr("class", "tick").attr("x", -6).attr("y", y(tt)).attr("dy", "0.35em")
            .attr("text-anchor", "end").text(pct(tt));
        }
      }
      if (!d) {
        g.append("text").attr("class", "tick").attr("x", plotW / 2).attr("y", plotH / 2).attr("text-anchor", "middle")
          .text(`too few composites (under ${votes.min_n})`);
      }
      g.append("line").attr("x1", 0).attr("x2", plotW).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", "#b9b8b2");
      const diag = g.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(top)).attr("y2", y(top))
        .attr("stroke", "#8a8984").attr("stroke-dasharray", "4 4");
      if (mode !== "end") diag.attr("x2", x(0)).attr("y2", y(0));
      if (mode === "play") tr(diag, t0, 700, d3.easeCubicInOut).attr("x2", x(top)).attr("y2", y(top));
      g.append("text").attr("class", "axis-title").attr("x", plotW / 2).attr("y", plotH + 33)
        .attr("text-anchor", "middle").text("share of the frame");
      if (k === 0 || narrow) {
        g.append("text").attr("class", "axis-title").attr("transform", `translate(${-42},${plotH / 2}) rotate(-90)`)
          .attr("text-anchor", "middle").text("share of the lead");
      }

      if (!d) return;
      const pts = CONCEPTS.map((c) => d.folded.find((r) => r.concept === c));
      const key = (p) => `${hab}|${p.concept}`;
      const dots = g.selectAll("circle").data(pts).join("circle")
        .attr("cx", (p) => x(p.area)).attr("cy", (p) => y(p.share)).attr("r", 5.5)
        .attr("fill", (p) => C[p.concept]).attr("stroke", "#fff").attr("stroke-width", 1.5)
        .on("mousemove", (event, p) => tip.show(event, `<b>${p.concept}</b> · ${hab}` +
          (p.parts ? `<br><span style="color:#8a8984">${p.parts.join(", ")}</span>` : "") +
          `<br>${pct(p.area)} of the frame<br>${d3.format("+.2f")(p.votes)} logits = ${pct(p.share)} of the lead`))
        .on("mouseleave", tip.hide);
      if (mode !== "end") dots.attr("cx", x(0)).attr("cy", y(0)).attr("opacity", 0);
      if (mode === "play") {
        tr(dots, (_, i) => t0 + 500 + i * 90, 800, d3.easeBackOut.overshoot(1.2))
          .attr("cx", (p) => x(p.area)).attr("cy", (p) => y(p.share)).attr("opacity", 1);
      }
      if (glide) {
        dots.filter((p) => prev.has(key(p)))
          .attr("cx", (p) => x(prev.get(key(p)).area)).attr("cy", (p) => y(prev.get(key(p)).share))
          .call((sel) => tr(sel, 0, 650, d3.easeCubicInOut).attr("cx", (p) => x(p.area)).attr("cy", (p) => y(p.share)));
      }
      pts.forEach((p) => prev.set(key(p), { area: p.area, share: p.share }));
      if (mode === "play" && k === 0) {
        // the trees: the biggest part of the frame and of the vote; ringed for a moment
        const tree = pts.find((p) => p.concept === "tree");
        const ring = g.append("circle").attr("cx", x(tree.area)).attr("cy", y(tree.share)).attr("r", 30)
          .attr("fill", "none").attr("stroke", "currentColor").attr("stroke-width", 2).attr("opacity", 0)
          .style("pointer-events", "none");
        tr(ring, t0 + 1700, 500, d3.easeBackOut).attr("r", 14).attr("opacity", 1)
          .transition().delay(1600).duration(600).attr("opacity", 0).remove();
      }
      const labels = g.append("g");
      if (mode !== "end" || glide) labels.attr("opacity", 0);
      if (mode === "play") tr(labels, t0 + 1300, 500).attr("opacity", 1);
      if (glide) tr(labels, 550, 300).attr("opacity", 1);

      // direct labels for the larger points, nudged apart vertically
      const labelled = pts.filter((p) => p.area >= 0.1 || Math.abs(p.share) >= 0.1)
        .map((p) => ({ p, ly: y(p.share) })).sort((a, b) => a.ly - b.ly);
      for (let i = 1; i < labelled.length; i++) {
        labelled[i].ly = Math.max(labelled[i].ly, labelled[i - 1].ly + 12);
      }
      for (const { p, ly } of labelled) {
        const left = x(p.area) > plotW - 60;
        if (Math.abs(ly - y(p.share)) > 4) { // moved away from its point: a short leader line
          labels.append("line").attr("x1", x(p.area)).attr("y1", y(p.share)).attr("x2", x(p.area) + (left ? -8 : 8))
            .attr("y2", ly).attr("stroke", "#b9b8b2");
        }
        labels.append("text").attr("class", "pt-label").attr("x", x(p.area) + (left ? -9 : 9)).attr("y", ly)
          .attr("dy", "0.35em").attr("text-anchor", left ? "end" : "start").text(p.concept);
      }
    });
  }
  const anim = introAnimation(fig, draw);
  onResize(draw);
}
