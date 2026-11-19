// Entry point: load data/*.json (written by phase2/pair_b/src/export_story_data.py), fill the numbers in the text,
// draw the figures and connect them to the scroll steps.

import { fillNumbers } from "./format.js";
import { initHero } from "./hero.js";
import { initRecap } from "./recap.js";
import { initEvidence } from "./evidence.js";
import { initUnits } from "./units.js";
import { initVotes } from "./votes.js";
import { initAblation } from "./ablation.js";
import { setupKeys, openBoxesForLinks } from "./keys.js";
import { setupScrolly } from "./scrolly.js";

const FILES = ["hero", "recap", "meta", "evidence", "units", "votes", "ablation"];

function notice(html) {
  const el = document.getElementById("notice");
  el.innerHTML = html;
  el.hidden = false;
}

openBoxesForLinks();

async function main() {
  const loaded = await Promise.all(FILES.map((f) => d3.json(`data/${f}.json`)));
  const data = Object.fromEntries(FILES.map((f, i) => [f, loaded[i]]));

  fillNumbers(document, data);
  const hero = initHero(data.hero);
  const recap = initRecap(data.recap, data.meta);
  initEvidence(data.evidence);
  initUnits(data.units);
  initVotes(data.votes);
  initAblation(data.ablation);

  setupScrolly("hero", (_, el) => hero.show(+el.dataset.frame), { offset: 0.6 });
  setupScrolly("recap", (_, el) => recap.show(el.dataset), { offset: 0.55 });
  setupKeys();
}

main().catch((err) => {
  console.error(err);
  notice(`Could not load the data (<code>data/*.json</code>): ${err.message}. Run
    <code>python phase2/pair_b/src/export_story_data.py</code>, then serve this folder, e.g.
    <code>python -m http.server 8000 --directory phase2/pair_b/story</code>.`);
});
