// Chapter 4: who votes for the great argus? Per habitat, every concept's share of the winner's lead (its "votes")
// against its share of the frame, over the wrong composites. On the diagonal a concept votes in proportion to its
// area. A toggle picks the winning class: all wrong answers (default) or one frequent winner, e.g. the great argus.

import { CONCEPT_COLOR, HABITAT_COLOR, onResize, pct, tooltip } from "./format.js";
import { toggle } from "./controls.js";

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
      winner = w;
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

    PANELS.forEach((hab, k) => {
      const d = panel(hab);
      const g = svg.append("g").attr("transform", narrow
        ? `translate(${m.left},${k * ph + m.top})` : `translate(${k * (pw + gap) + m.left},${m.top})`);
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
      g.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(top)).attr("y2", y(top))
        .attr("stroke", "#8a8984").attr("stroke-dasharray", "4 4");
      g.append("text").attr("class", "axis-title").attr("x", plotW / 2).attr("y", plotH + 33)
        .attr("text-anchor", "middle").text("share of the frame");
      if (k === 0 || narrow) {
        g.append("text").attr("class", "axis-title").attr("transform", `translate(${-42},${plotH / 2}) rotate(-90)`)
          .attr("text-anchor", "middle").text("share of the lead");
      }

      if (!d) return;
      const pts = CONCEPTS.map((c) => d.folded.find((r) => r.concept === c));
      g.selectAll("circle").data(pts).join("circle")
        .attr("cx", (p) => x(p.area)).attr("cy", (p) => y(p.share)).attr("r", 5.5)
        .attr("fill", (p) => C[p.concept]).attr("stroke", "#fff").attr("stroke-width", 1.5)
        .on("mousemove", (event, p) => tip.show(event, `<b>${p.concept}</b> · ${hab}` +
          (p.parts ? `<br><span style="color:#8a8984">${p.parts.join(", ")}</span>` : "") +
          `<br>${pct(p.area)} of the frame<br>${d3.format("+.2f")(p.votes)} logits = ${pct(p.share)} of the lead`))
        .on("mouseleave", tip.hide);

      // direct labels for the larger points, nudged apart vertically
      const labelled = pts.filter((p) => p.area >= 0.1 || Math.abs(p.share) >= 0.1)
        .map((p) => ({ p, ly: y(p.share) })).sort((a, b) => a.ly - b.ly);
      for (let i = 1; i < labelled.length; i++) {
        labelled[i].ly = Math.max(labelled[i].ly, labelled[i - 1].ly + 12);
      }
      for (const { p, ly } of labelled) {
        const left = x(p.area) > plotW - 60;
        if (Math.abs(ly - y(p.share)) > 4) { // moved away from its point: a short leader line
          g.append("line").attr("x1", x(p.area)).attr("y1", y(p.share)).attr("x2", x(p.area) + (left ? -8 : 8))
            .attr("y2", ly).attr("stroke", "#b9b8b2");
        }
        g.append("text").attr("class", "pt-label").attr("x", x(p.area) + (left ? -9 : 9)).attr("y", ly)
          .attr("dy", "0.35em").attr("text-anchor", left ? "end" : "start").text(p.concept);
      }
    });
  }
  onResize(draw);
}
