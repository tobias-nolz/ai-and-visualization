// Chapter 1: Phase 1 recap in one sticky figure with three views:
//   map      territory dot map (x = camera openness, y = frame brightness, rows = day / greyscale), as in
//            phase1/pair_b/notebooks/confusion_territories.ipynb; colour = habitat or the majority outcome of a view
//   controls share correct per species group: grey frame, own camera, other cameras
//   cells    share predicted "empty" per paste cell: other cameras, grey frame, own camera
// The map has a picker for the animal and the paste position (data/map/<group>.json, loaded when picked); a scroll
// step resets it to the step's view.

import {
  HABITATS, HABITAT_COLOR, INK, MUTED, OUTCOME_COLOR, OUTCOME_LABEL, pct,
} from "./format.js";
import { dropdown } from "./controls.js";

const LIGHTS = ["day", "greyscale"]; // Phase 1 "night" = greyscale frames (infrared or B/W cameras)
const BANDS = { forest: [-Infinity, 1 / 3], mixed: [1 / 3, 2 / 3], open: [2 / 3, Infinity] }; // SegFormer thresholds
const GREY_MARK = "#8a8984";
const ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"; // answers in data/map/*.json
const ROW_NAMES = ["top", "middle", "bottom"];
const COL_NAMES = ["left", "centre", "right"];

export function initRecap(recap, meta) {
  const fig = d3.select("#recap-figure");
  const body = fig.select(".fig-body");
  const tip = body.select(".tooltip");
  const svgs = {
    map: body.select("svg.view-map"),
    controls: body.select("svg.view-controls"),
    cells: body.select("svg.view-cells"),
  };
  const frames = recap.map.frames;
  const views = recap.map.views;
  let state = { view: "map", map: "elephant" }; // "habitat" (colour = habitat) is still supported
  let picked = null; // a view from the picker (data/map/*.json), shown instead of views[state.map]
  const current = () => picked || views[state.map];

  // ------------------------------------------------------------------ figure head per state
  const HEADS = {
    habitat: () => ({
      title: `${meta.backgrounds} empty frames from ${meta.cameras} cameras`,
      sub: "x: openness of the camera (SegFormer) · y: brightness of the frame · colour: habitat",
      legend: HABITATS.map((h) => ({ label: h, color: HABITAT_COLOR[h], sq: true })),
      note: "Frames of one camera share its openness (slightly jittered so they don't overlap).",
    }),
    outcome: (v) => ({
      title: `${v.species}, pasted at the ${v.cell}`,
      sub: "Colour: the model's most common answer over the three cut-outs on each frame. Hover a dot for all three.",
      legend: OUTCOME_LABEL.map((label, i) => ({ label, color: OUTCOME_COLOR[i] })),
      note: `${d3.format(",")(v.n)} composites. Numbers above each band: share correct and the most common wrong answer.`,
    }),
    controls: () => ({
      title: "Named correctly, per species group",
      sub: `Grey frame: ${recap.controls.overall.grey_n_per_group} composites per group (3 cut-outs × 9 cells). ` +
        "Own camera: empty frames from the camera the cut-out was photographed at. Other cameras: all composites.",
      legend: [
        { label: "grey frame", color: GREY_MARK, sq: true },
        { label: "own camera", ring: true },
        { label: "other cameras (the composites)", color: OUTCOME_COLOR[0] },
      ],
      note: "Sorted by the grey-frame share. Hover a row for the numbers.",
    }),
    cells: () => ({
      title: "Predicted empty, per paste position",
      sub: "Share of composites the model calls “empty”, for each cell of the 3 × 3 grid the cut-out was pasted on.",
      legend: [{ label: "share predicted empty", color: OUTCOME_COLOR[3], sq: true }],
      note: "",
    }),
  };

  function setHead(h) {
    fig.select(".fig-title").text(h.title);
    fig.select(".fig-sub").text(h.sub);
    fig.select(".fig-note").text(h.note);
    const items = fig.select(".legend").selectAll(".item").data(h.legend).join("span").attr("class", "item");
    items.html("");
    items.append("i")
      .attr("class", (d) => (d.sq ? "sq" : d.ring ? "ring" : null))
      .style("background", (d) => d.color || null);
    items.append("span").text((d) => d.label);
  }

  // ------------------------------------------------------------------ tooltip
  function showTip(event, html) {
    const box = body.node().getBoundingClientRect();
    const x = event.clientX - box.left;
    const y = event.clientY - box.top;
    tip.html(html).classed("is-on", true);
    const t = tip.node().getBoundingClientRect();
    const left = x + 14 + t.width > box.width ? x - t.width - 14 : x + 14;
    tip.style("left", `${Math.max(0, left)}px`).style("top", `${Math.max(0, y - t.height - 10)}px`);
  }
  const hideTip = () => tip.classed("is-on", false);

  // ------------------------------------------------------------------ view: territory map
  let dots = null;
  let bandStats = null;

  function drawMap(W, H) {
    const svg = svgs.map.attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const m = { top: 22, right: 8, bottom: 46, left: 64 };
    const head = 32; // per row: share correct + most common wrong answer above the band
    const gap = 10;
    const rowH = (H - m.top - m.bottom - 2 * head - gap) / 2;
    const rowTop = { day: m.top + head, greyscale: m.top + head + rowH + gap + head };
    const lo = Math.min(0, d3.min(frames, (d) => d.x)) - 0.03;
    const hi = Math.max(1, d3.max(frames, (d) => d.x)) + 0.03;
    const x = d3.scaleLinear().domain([lo, hi]).range([m.left, W - m.right]);
    const y = (d) => rowTop[d.light] + 5 + (1 - d.y) * (rowH - 10);
    const bandX = (h) => [x(Math.max(lo, BANDS[h][0])), x(Math.min(hi, BANDS[h][1]))];

    const g = svg.append("g");
    for (const light of LIGHTS) {
      for (const h of HABITATS) {
        const [x0, x1] = bandX(h);
        g.append("rect")
          .attr("x", x0).attr("width", x1 - x0)
          .attr("y", rowTop[light]).attr("height", rowH)
          .attr("fill", HABITAT_COLOR[h]).attr("fill-opacity", 0.09);
      }
      g.append("text").attr("class", "row-label")
        .attr("transform", `translate(14,${rowTop[light] + rowH / 2}) rotate(-90)`)
        .attr("text-anchor", "middle").text(light.toUpperCase());
      g.append("text").attr("class", "tick").attr("x", m.left - 8).attr("y", rowTop[light] + 10)
        .attr("text-anchor", "end").text("bright");
      g.append("text").attr("class", "tick").attr("x", m.left - 8).attr("y", rowTop[light] + rowH - 3)
        .attr("text-anchor", "end").text("dark");
    }
    for (const h of HABITATS) {
      const [x0, x1] = bandX(h);
      g.append("text").attr("class", "band-label").attr("x", (x0 + x1) / 2).attr("y", 12)
        .attr("text-anchor", "middle").attr("fill", HABITAT_COLOR[h]).style("fill", HABITAT_COLOR[h]).text(h);
    }
    // openness axis
    const axisY = rowTop.greyscale + rowH;
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      g.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", axisY).attr("y2", axisY + 4).attr("stroke", MUTED);
      g.append("text").attr("class", "tick").attr("x", x(t)).attr("y", axisY + 17).attr("text-anchor", "middle")
        .text(`${t * 100}%`);
    }
    g.append("text").attr("class", "axis-title").attr("x", (x(0) + x(1)) / 2).attr("y", axisY + 38)
      .attr("text-anchor", "middle").text("openness of the camera = share of the background without trees →");

    // per band: two short lines above it (filled in by colourMap)
    bandStats = g.append("g").selectAll("g")
      .data(LIGHTS.flatMap((light) => HABITATS.map((h) => ({ light, h }))))
      .join("g")
      .attr("transform", (d) => `translate(${bandX(d.h)[0] + 4},${rowTop[d.light] - head + 13})`);
    bandStats.append("text").attr("class", "stat");
    bandStats.append("text").attr("class", "stat-sub").attr("y", 14);

    const r = Math.max(3, Math.min(5, W / 160));
    dots = svg.append("g").selectAll("circle")
      .data(frames)
      .join("circle")
      .attr("class", "dot")
      .attr("cx", (d) => x(d.x))
      .attr("cy", y)
      .attr("r", r)
      .on("mousemove", (event, d) => showTip(event, mapTip(d)))
      .on("mouseleave", hideTip);
    dots.each(function (d) { d.px = x(d.x); }); // for the colour sweep
    colourMap(false);
  }

  function mapTip(d) {
    const i = frames.indexOf(d);
    let html = `<b>camera ${d.cam}</b> · ${d.hab} · ${d.light}<br>openness ${pct(d.x < 0 ? 0 : d.x)}, ` +
      `brightness ${pct(d.y)}`;
    if (state.map !== "habitat") {
      const v = current();
      html += `<br>answers for the three ${v.species} cut-outs:` + v.answers[i].map(([s, o]) =>
        `<span class="ans"><i style="background:${OUTCOME_COLOR[o]}"></i>${recap.map.species[s]}</span>`).join("");
    }
    return html;
  }

  function colourMap(animate = true) {
    if (!dots) return;
    const key = state.map;
    const color = key === "habitat"
      ? (d) => HABITAT_COLOR[d.hab]
      : (d, i) => OUTCOME_COLOR[current().outcome[i]];
    const W = +svgs.map.attr("viewBox").split(" ")[2];
    (animate ? dots.transition().duration(650).delay((d) => (d.px / W) * 450) : dots).attr("fill", color);

    bandStats.each(function ({ light, h }) {
      const g = d3.select(this);
      if (key === "habitat") {
        const c = meta.per_habitat[`${light}|${h}`];
        g.select(".stat").text(`${c.frames} frames`);
        g.select(".stat-sub").text(`${c.cameras} cameras`);
      } else {
        const s = current().habitats[`${light}|${h}`];
        g.select(".stat").text(`${pct(s.correct)} correct`);
        g.select(".stat-sub").text(s.wrong ? `${s.wrong} ${pct(s.wrong_share)}` : "");
      }
    });
  }

  // ------------------------------------------------------------------ view: controls (dot plot)
  function drawControls(W, H) {
    const svg = svgs.controls.attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const rows = recap.controls.groups;
    const m = { top: 8, right: 18, bottom: 46, left: 118 };
    const rowH = Math.min(36, (H - m.top - m.bottom) / rows.length);
    const x = d3.scaleLinear().domain([0, 1]).range([m.left, W - m.right]);
    const y = (i) => m.top + (i + 0.5) * rowH;
    const bottom = m.top + rows.length * rowH;

    const g = svg.append("g");
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      g.append("line").attr("class", "gridline").attr("x1", x(t)).attr("x2", x(t)).attr("y1", m.top).attr("y2", bottom);
      g.append("text").attr("class", "tick").attr("x", x(t)).attr("y", bottom + 16).attr("text-anchor", "middle")
        .text(`${t * 100}%`);
    }
    g.append("text").attr("class", "axis-title").attr("x", (x(0) + x(1)) / 2).attr("y", bottom + 36)
      .attr("text-anchor", "middle").text("share named correctly (exact species)");

    const row = g.selectAll("g.row").data(rows).join("g").attr("class", "row")
      .attr("transform", (_, i) => `translate(0,${y(i)})`);
    row.append("rect").attr("x", 0).attr("width", W).attr("y", -rowH / 2).attr("height", rowH)
      .attr("fill", "transparent");
    row.append("text").attr("class", "grp").attr("x", m.left - 14).attr("dy", "0.35em").attr("text-anchor", "end")
      .text((d) => d.group);
    row.append("line")
      .attr("x1", (d) => x(Math.min(d.grey, d.own, d.swap)))
      .attr("x2", (d) => x(Math.max(d.grey, d.own, d.swap)))
      .attr("stroke", "#d6d5cf").attr("stroke-width", 2);
    // own camera first, so a grey square at the same value stays visible on top of the ring
    row.append("circle").attr("cx", (d) => x(d.own)).attr("r", 5).attr("fill", "#fff")
      .attr("stroke", INK).attr("stroke-width", 1.7);
    const s = 9;
    row.append("rect").attr("x", (d) => x(d.grey) - s / 2).attr("y", -s / 2).attr("width", s).attr("height", s)
      .attr("fill", GREY_MARK);
    row.append("circle").attr("cx", (d) => x(d.swap)).attr("r", 5.5).attr("fill", OUTCOME_COLOR[0])
      .attr("stroke", "#fff").attr("stroke-width", 1);
    row.on("mousemove", (event, d) => showTip(event,
      `<b>${d.group}</b><br>grey frame: ${pct(d.grey)} of ${d.grey_n}<br>own camera: ${pct(d.own)} of ` +
      `${d3.format(",")(d.own_n)}<br>other cameras: ${pct(d.swap)} of ${d3.format(",")(d.swap_n)}`))
      .on("mouseleave", hideTip);
  }

  // ------------------------------------------------------------------ view: cells (3 x 3 grids)
  function drawCells(W, H) {
    const svg = svgs.cells.attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const c = recap.cells;
    const ROW = ["top row", "middle row", "bottom row"];
    // the composites big on top (with row shares on the right), the two controls small below
    const rowLabelW = 118;
    const mw = Math.min(W - rowLabelW - 10, ((H - 70) * 0.58) / 0.75);
    const mh = mw * 0.75;
    const sw = (mw - 24) / 2;
    const sh = sw * 0.75;
    const grids = [
      { key: "swap", title: "Other cameras' frames", x: 0, y: 22, w: mw, h: mh, rows: true },
      { key: "grey", title: "Grey frame", x: 0, y: 22 + mh + 52, w: sw, h: sh },
      { key: "own", title: "Own camera", x: sw + 24, y: 22 + mh + 52, w: sw, h: sh },
    ];
    for (const gr of grids) {
      const d = c[gr.key];
      const g = svg.append("g").attr("transform", `translate(${gr.x},${gr.y})`);
      g.append("text").attr("class", "grid-title").attr("y", -8)
        .text(`${gr.title}`)
        .append("tspan").attr("class", "val").attr("dx", 8).text(`${d3.format(",")(d.n)} composites`);
      const cw = gr.w / 3;
      const ch = gr.h / 3;
      d.cells.forEach((rowVals, r) => rowVals.forEach((v, col) => {
        const cell = g.append("g").attr("transform", `translate(${col * cw},${r * ch})`);
        cell.append("rect").attr("width", cw).attr("height", ch).attr("fill", "#fff")
          .attr("stroke", "#dddcd6");
        cell.append("rect").attr("x", 1).attr("width", cw - 2).attr("y", ch - (ch - 1) * v).attr("height", (ch - 1) * v)
          .attr("fill", OUTCOME_COLOR[3]);
        cell.append("text").attr("class", "cell-val").attr("x", cw / 2).attr("y", gr.rows ? 22 : 15)
          .attr("text-anchor", "middle").style("font-size", gr.rows ? "15px" : "12px").text(pct(v));
      }));
      g.append("rect").attr("width", gr.w).attr("height", gr.h).attr("fill", "none").attr("stroke", "#b9b8b2");
      if (gr.rows) {
        d.rows.forEach((v, r) => {
          const t = g.append("text").attr("class", "stat").attr("x", gr.w + 12).attr("y", (r + 0.5) * ch)
            .attr("dy", "0.35em");
          t.text(pct(v));
          t.append("tspan").attr("class", "stat-sub").attr("dx", 5).text(ROW[r]);
        });
      }
    }
  }

  // ------------------------------------------------------------------ state + resize
  function render() {
    const { width, height } = body.node().getBoundingClientRect();
    if (width < 50 || height < 50) return;
    drawMap(width, height);
    drawControls(width, height);
    drawCells(width, height);
  }

  /** A scroll step: its view; the picker goes back to the step's animal and position. */
  function show({ view, map }) {
    const prev = { ...state, picked };
    state = { view, map: map || state.map };
    picked = null;
    apply(prev.view === "map" && (prev.map !== state.map || prev.picked !== null));
  }

  function apply(animate) {
    const { view } = state;
    for (const [k, svg] of Object.entries(svgs)) svg.classed("is-active", k === view);
    hideTip();
    if (view === "map") {
      setHead(state.map === "habitat" ? HEADS.habitat() : HEADS.outcome(current()));
      colourMap(animate);
    } else {
      setHead(HEADS[view]());
    }
    syncPicker();
  }

  // ------------------------------------------------------------------ picker: animal and paste position
  const files = new Map(); // group -> Promise of its data/map file
  const controls = d3.select("#recap-controls");
  const animal = dropdown(controls, {
    label: "Animal",
    options: recap.map.groups.map((g) => ({ value: g.group, label: g.species })),
    value: "elephant",
    onChange: (group) => pick(group, `${current().row}${current().col}`),
  });
  const padWrap = controls.append("div").attr("class", "ctl");
  padWrap.append("span").attr("class", "ctl-label").text("Pasted at");
  const pad = padWrap.append("div").attr("class", "pad").attr("role", "group").attr("aria-label", "Paste position");
  const cells = d3.range(9).map((i) => ({ row: Math.floor(i / 3), col: i % 3 }));
  const padButtons = pad.selectAll("button").data(cells).join("button").attr("type", "button")
    .attr("aria-label", (d) => `${ROW_NAMES[d.row]} ${COL_NAMES[d.col]}`)
    .attr("title", (d) => `${ROW_NAMES[d.row]} ${COL_NAMES[d.col]}`)
    .on("click", function (event, d) {
      if (event.detail > 0) this.blur(); // a mouse click: keep the arrow keys / space for the beats
      pick(current().group, `${d.row}${d.col}`);
    });

  function syncPicker() {
    controls.style("visibility", state.view === "map" && state.map !== "habitat" ? null : "hidden");
    const v = current();
    if (!v || state.map === "habitat") return;
    animal.set(v.group);
    padButtons.attr("aria-pressed", (d) => String(d.row === v.row && d.col === v.col));
  }

  function decode(v) {
    const answers = frames.map((_, i) => [0, 1, 2].map((j) => v.codes[ALPHABET.indexOf(v.answers[3 * i + j])]));
    // dot colour: the most common outcome of the three cut-outs, ties go to the better outcome (as in the export)
    const outcome = answers.map((a) => {
      const counts = [0, 0, 0, 0];
      for (const [, o] of a) counts[o] += 1;
      return counts.indexOf(Math.max(...counts));
    });
    return { ...v, answers, outcome };
  }

  async function pick(group, cell) {
    const entry = recap.map.groups.find((g) => g.group === group);
    if (!files.has(group)) files.set(group, d3.json(`data/${entry.file}`));
    const file = await files.get(group);
    if (state.view !== "map") return; // scrolled on meanwhile
    picked = decode(file[cell]);
    apply(true);
  }

  render();
  apply(false);
  let timer;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(() => { render(); apply(false); }, 150);
  });
  return { show };
}
