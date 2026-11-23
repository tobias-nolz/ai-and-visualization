// Takeaway: how much of the frame is the animal? 100 squares, one per percent of the frame, for the average wrong
// composite: the cut-out's share in blue (the colour of evidence for the true species in ch. 2), the background in
// orange (evidence for the winning class). Intro: the background fills in square by square, then the animal.

import { TRUE_COLOR, WINNER_COLOR, pct } from "./format.js";
import { introAnimation, tr } from "./reveal.js";

export function initTakeaway(ev) {
  const fig = d3.select("#waffle-figure");
  const share = ev.wrong.mask_area;
  const nAnimal = Math.round(share * 100);
  const s = 30, gap = 4, step = s + gap, side = 10 * step - gap;
  const W = side, H = side + 62;
  const squares = d3.range(100).map((k) => ({ k, r: Math.floor(k / 10), c: k % 10, animal: k >= 100 - nAnimal }));

  function draw() {
    const mode = anim.mode;
    const svg = fig.select("svg").attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const rects = svg.selectAll("rect").data(squares).join("rect")
      .attr("x", (d) => d.c * step).attr("y", (d) => d.r * step).attr("width", s).attr("height", s).attr("rx", 3)
      .attr("fill", (d) => (d.animal ? TRUE_COLOR : WINNER_COLOR))
      .attr("fill-opacity", (d) => (d.animal ? 1 : 0.85));
    const labels = svg.append("g").attr("font-size", 17).attr("font-weight", 700).attr("text-anchor", "middle");
    labels.append("text").attr("x", W / 2).attr("y", side + 26).style("fill", TRUE_COLOR)
      .text(`animal: ${pct(share)} of the frame`);
    labels.append("text").attr("x", W / 2).attr("y", side + 50).style("fill", WINNER_COLOR)
      .text(`background: ${pct(1 - share)}`);
    if (mode === "end") return;
    rects.attr("opacity", 0);
    labels.attr("opacity", 0);
    if (mode !== "play") return;
    tr(rects.filter((d) => !d.animal), (d) => 100 + d.k * 14, 300).attr("opacity", 1);
    tr(rects.filter((d) => d.animal), (d) => 1800 + (d.k - (100 - nAnimal)) * 120, 500, d3.easeBackOut)
      .attr("opacity", 1);
    tr(labels, 2500, 500).attr("opacity", 1);
  }

  const anim = introAnimation(fig, draw, { threshold: 0.5 });
  draw();
}
