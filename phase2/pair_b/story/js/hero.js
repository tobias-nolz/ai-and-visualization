// Chapter 0: the same elephant cut-out on an open and a forest frame; cross-fades with the steps.
// The frames differ in aspect ratio (16:9 vs 4:3): the stage takes the narrowest one, so both pictures fill the
// same box; wider pictures lose a strip at each side (the cut-out sits at the bottom centre).

import { HABITAT_COLOR, OUTCOMES, OUTCOME_COLOR, pct } from "./format.js";

const VERDICT = {
  "exact species": "correct",
  "same group": "wrong: right group, other species",
  "other group": "wrong: other group",
  empty: "wrong: predicted empty",
};

export function initHero(hero) {
  const stage = d3.select("#hero-stage");
  const narrowest = d3.least(hero.frames, (d) => d.width / d.height);
  d3.select(stage.node().parentNode).style("--ar", narrowest.width / narrowest.height); // .hero-figure
  const imgs = stage
    .selectAll("img")
    .data(hero.frames)
    .join("img")
    .attr("src", (d) => `img/${d.img}`)
    .attr("width", (d) => d.width)
    .attr("height", (d) => d.height)
    .attr("decoding", "async")
    .attr("alt", (d) => `${hero.species} cut-out pasted at the ${hero.cell} of an empty daytime frame from camera ` +
      `${d.camera} (${d.habitat}). The model's answer: ${d.answer}, ${pct(d.confidence)}.`);

  const label = d3.select(".hero-label");
  const where = d3.select(".hero-where");

  function show(i) {
    const d = hero.frames[i];
    const color = OUTCOME_COLOR[OUTCOMES.indexOf(d.outcome)];
    imgs.classed("is-active", (_, j) => j === i);
    label.select(".hero-answer .txt").text(d.answer);
    label.select(".hero-answer .dot").style("background", color);
    label.select(".hero-conf").text(pct(d.confidence));
    label.select(".hero-bar i").style("width", `${d.confidence * 100}%`).style("background", color);
    where.select(".chip").style("--chip", HABITAT_COLOR[d.habitat]).text(`${d.habitat} · day`);
    where.select(".where-txt").text(
      `camera ${d.camera} (${d.split}), openness ${pct(d.camera_openness)} · ${VERDICT[d.outcome]}`,
    );
  }

  show(0);
  return { show };
}
