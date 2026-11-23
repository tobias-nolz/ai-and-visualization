/* Inside Camera 156 · Phase 2 Pair B White-Box Explorer (Option B) */
(() => {
  "use strict";

  const DATA_URL = "/results/camera156_whitebox.json";

  const state = {
    data: null,
    cohort: "all",       // "all" | "night" | "day"
    mode: "baseline",    // "baseline" | "cover_bar" | "zero_r13" | "zero_r12_13" | "zero_r6_7"
    knnBackbone: "ft",   // "ft" | "frozen"
    knnScope: "pred",    // "pred" (ExMatchina predicted class) | "all" (unfiltered)
    frameIdx: 0,
    prevAllFrameIdx: 0,
    hoverStrip: null,
    stripVbW: 196,
  };

  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs = {}, parent = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== null && v !== undefined) node.setAttribute(k, String(v));
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function fmtInt(n) {
    return Number(n).toLocaleString("en-US");
  }

  function fmtPct(v, dec = 1) {
    return `${Number(v).toFixed(dec)}%`;
  }

  function fmtSigned(v, dec = 2) {
    const n = Number(v);
    return `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(dec)}`;
  }

  function fmtSpeciesLabel(common, sci) {
    if (!sci || common.toLowerCase() === sci.toLowerCase()) return common;
    return `${common} (${sci})`;
  }

  // Tooltip helpers
  const tooltip = $("tooltip");

  function showTooltip(ev, html) {
    tooltip.innerHTML = html;
    tooltip.style.display = "block";
    moveTooltip(ev);
  }

  function moveTooltip(ev) {
    if (tooltip.style.display !== "block") return;
    const pad = 12;
    const rect = tooltip.getBoundingClientRect();
    let x = ev.clientX + pad;
    let y = ev.clientY + pad;
    if (x + rect.width > window.innerWidth - 8) x = ev.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8) y = ev.clientY - rect.height - pad;
    tooltip.style.left = `${Math.max(6, x)}px`;
    tooltip.style.top = `${Math.max(6, y)}px`;
  }

  function hideTooltip() {
    tooltip.style.display = "none";
  }

  // Exact object-fit:contain bounding box placement (0.0px drift at any aspect ratio or resize)
  function placeBbox(boxEl, imgEl, bbox) {
    if (!boxEl || !imgEl || !Array.isArray(bbox) || bbox.length !== 4) {
      if (boxEl) boxEl.style.display = "none";
      return;
    }
    const parent = boxEl.parentElement;
    const nw = imgEl.naturalWidth;
    const nh = imgEl.naturalHeight;
    if (!parent || !nw || !nh) {
      boxEl.style.display = "none";
      return;
    }
    const cw = parent.clientWidth;
    const ch = parent.clientHeight;
    if (!cw || !ch) return;
    const scale = Math.min(cw / nw, ch / nh);
    const rw = nw * scale;
    const rh = nh * scale;
    const ox = (cw - rw) / 2;
    const oy = (ch - rh) / 2;
    const [x0, y0, x1, y1] = bbox;
    boxEl.style.left = `${ox + x0 * rw}px`;
    boxEl.style.top = `${oy + y0 * rh}px`;
    boxEl.style.width = `${Math.max(4, (x1 - x0) * rw)}px`;
    boxEl.style.height = `${Math.max(4, (y1 - y0) * rh)}px`;
    boxEl.style.display = "block";
  }

  let activeDialogBbox = null;

  // Click-to-zoom dialog (clears prior src immediately to prevent stale-frame flash)
  const imageDialog = $("imageDialog");
  $("dialogClose").addEventListener("click", () => imageDialog.close());
  imageDialog.addEventListener("click", (ev) => {
    if (ev.target === imageDialog) imageDialog.close();
  });

  function openZoom(title, imgUrl, caption, bbox = null) {
    const img = $("dialogImg");
    const box = $("dialogBbox");
    activeDialogBbox = bbox;
    box.style.display = "none";
    img.onload = () => placeBbox(box, img, activeDialogBbox);
    img.removeAttribute("src");
    $("dialogTitle").textContent = title;
    $("dialogCaption").textContent = caption;
    imageDialog.showModal();
    img.src = imgUrl;
    if (img.complete && img.naturalWidth) placeBbox(box, img, activeDialogBbox);
  }

  // =========================================================================
  // EXACT 1:1 SQUARE PHOTO + 0 PX VERTICAL ALIGNMENT FOR 18-STRIP CHART
  // =========================================================================
  function syncSpatialSquare() {
    const pair = $("spatialPair");
    const photoBox = $("photoBox");
    const stripCol = $("stripCol");
    if (!pair || !photoBox || !stripCol) return;

    const availW = pair.clientWidth;
    const availH = pair.clientHeight;
    if (availW <= 0 || availH <= 0) return;

    const gap = 10;
    const minStripW = 166;
    const maxStripW = 224;
    const side = Math.max(180, Math.floor(Math.min(availH - 2, availW - minStripW - gap)));
    const stripW = Math.max(minStripW, Math.min(maxStripW, Math.floor(availW - side - gap)));

    photoBox.style.width = `${side}px`;
    photoBox.style.height = `${side}px`;
    stripCol.style.width = `${stripW}px`;
    stripCol.style.height = `${side}px`;

    // Match viewBox aspect ratio to rendered CSS box so SVG text/markers never distort
    // AND y=0..448 maps identically pixel-for-pixel to the 448x448 photo box!
    const nextVbW = Math.max(165, Math.round((stripW / side) * 448));
    if (Math.abs(nextVbW - state.stripVbW) >= 2 && state.data) {
      state.stripVbW = nextVbW;
      renderStripChart();
    } else {
      state.stripVbW = nextVbW;
    }
  }

  // =========================================================================
  // COLUMN 1: LAYER 1..49 CONTRASTIVE GRAD-CAM + BASELINE COHORT BAR
  // =========================================================================
  function renderCol1() {
    const { data, cohort } = state;
    const traj = data.layer_trajectories[cohort];
    const stages = traj.stages;
    const meta = data.meta;

    $("col1ObjTag").textContent = `Target: ${traj.objective_label}`;

    const svg = $("layerChart");
    svg.innerHTML = "";
    const W = 408;
    const H = 286;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    const m = { t: 18, r: 58, b: 36, l: 38 };
    const pw = W - m.l - m.r;
    const ph = H - m.t - m.b;

    const xAt = (i) => m.l + (i / (stages.length - 1)) * pw;
    const yAt = (pct) => m.t + ph * (1 - Math.max(0, Math.min(100, pct)) / 100);

    // Gridlines & Y-axis
    for (const tick of [0, 25, 50, 75, 100]) {
      const y = yAt(tick);
      el("line", {
        x1: m.l, y1: y, x2: xAt(4) + 14, y2: y,
        stroke: tick === 0 ? "#bbb" : "#ececec",
        "stroke-width": 1,
      }, svg);
      const t = el("text", {
        x: m.l - 6, y: y + 3.5, "text-anchor": "end",
        "font-size": 10.5, fill: "#666",
      }, svg);
      t.textContent = `${tick}%`;
    }

    // Well-spaced Layer 50 (fc) vertical marker on far right (38 px past Layer 49 -> zero overlap!)
    const xFc = xAt(4) + 38;
    el("line", {
      x1: xFc, y1: m.t, x2: xFc, y2: m.t + ph,
      stroke: "#d4d4d4", "stroke-dasharray": "2,3", "stroke-width": 1,
    }, svg);
    const fcLbl = el("text", {
      x: xFc, y: m.t + ph + 16, "text-anchor": "middle",
      "font-size": 9.5, fill: "#888",
    }, svg);
    fcLbl.textContent = "L50";
    const fcSub = el("text", {
      x: xFc, y: m.t + ph + 27, "text-anchor": "middle",
      "font-size": 9, fill: "#999",
    }, svg);
    fcSub.textContent = "(fc)";

    // X-axis stage labels
    stages.forEach((s, i) => {
      const x = xAt(i);
      el("line", {
        x1: x, y1: m.t + ph, x2: x, y2: m.t + ph + 4,
        stroke: "#aaa", "stroke-width": 1,
      }, svg);
      const lbl = el("text", {
        x, y: m.t + ph + 16, "text-anchor": "middle",
        "font-size": 10.5, "font-weight": s.depth >= 40 ? "700" : "400",
        fill: s.depth >= 40 ? "#111" : "#555",
      }, svg);
      lbl.textContent = s.label;
      const sub = el("text", {
        x, y: m.t + ph + 27, "text-anchor": "middle",
        "font-size": 9, fill: "#888",
      }, svg);
      sub.textContent = `${s.grid}×${s.grid}`;
    });

    const linePath = (key) =>
      stages.map((s, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAt(s[key]).toFixed(1)}`).join(" ");

    // 1. Normal empty control (dotted gray)
    el("path", {
      d: linePath("ctrl_ctr_pct"),
      fill: "none", stroke: "#aaaaaa", "stroke-width": 1.6, "stroke-dasharray": "2,3",
    }, svg);

    // 2. Frozen ImageNet Linear Probe (faint dashed slate)
    el("path", {
      d: linePath("frozen_ctr_pct"),
      fill: "none", stroke: "#7a8694", "stroke-width": 1.9, "stroke-dasharray": "5,4",
    }, svg);

    // 3. Fine-Tuned ResNet-50 (bold gold/amber)
    el("path", {
      d: linePath("ft_ctr_pct"),
      fill: "none", stroke: "#806000", "stroke-width": 2.7,
    }, svg);

    // Hover guide line
    const guide = el("line", {
      x1: 0, y1: m.t, x2: 0, y2: m.t + ph,
      stroke: "#111", "stroke-width": 1, "stroke-dasharray": "2,2", opacity: 0,
    }, svg);

    const haloStyle = "paint-order:stroke;stroke:#ffffff;stroke-width:3.5px;stroke-linejoin:round;";

    // Points & clean haloed labels
    stages.forEach((s, i) => {
      const x = xAt(i);
      const yFt = yAt(s.ft_ctr_pct);
      const yFr = yAt(s.frozen_ctr_pct);
      const yCt = yAt(s.ctrl_ctr_pct);

      el("circle", { cx: x, cy: yCt, r: 2.4, fill: "#aaa" }, svg);
      el("circle", { cx: x, cy: yFr, r: 3.6, fill: "white", stroke: "#7a8694", "stroke-width": 1.7 }, svg);
      el("circle", { cx: x, cy: yFt, r: 4.6, fill: "#DDAA33", stroke: "#806000", "stroke-width": 1.8 }, svg);

      if (i >= 2) {
        const dy = -11;
        const lbl = el("text", {
          x, y: yFt + dy, "text-anchor": "middle",
          "font-size": 11, "font-weight": "700", fill: "#806000", style: haloStyle,
        }, svg);
        lbl.textContent = fmtPct(s.ft_ctr_pct, 1);
      }
      if (i === 4) {
        // Place Frozen Layer 49 label cleanly above or below its dot away from Ctrl line
        const placeAbove = s.frozen_ctr_pct >= s.ctrl_ctr_pct;
        const fLbl = el("text", {
          x: x + 4, y: yFr + (placeAbove ? -9 : 14), "text-anchor": "middle",
          "font-size": 10, "font-weight": "600", fill: "#5c6875", style: haloStyle,
        }, svg);
        fLbl.textContent = fmtPct(s.frozen_ctr_pct, 1);
      }

      const hitW = pw / (stages.length - 1);
      const hit = el("rect", {
        x: x - hitW / 2, y: m.t, width: hitW, height: ph,
        fill: "transparent", style: "cursor:help;",
      }, svg);
      hit.addEventListener("mouseenter", (ev) => {
        guide.setAttribute("x1", x);
        guide.setAttribute("x2", x);
        guide.setAttribute("opacity", "0.45");
        let extra = "";
        if (s.id === "layer4") {
          extra = `<div class="tt-sep"></div>
            <b>Layer 49 row breakdown (Fine-Tuned):</b><br>
            Row 13 only (7.1% area): <b>${fmtPct(s.ft_ctr_r13_pct)}</b><br>
            Rows 12–13 (14.3% area): <b>${fmtPct(s.ft_ctr_pct)}</b><br>
            Rows 11–13 (21.4% area): <b>${fmtPct(s.ft_ctr_r11_13_pct)}</b><br>
            Frozen ImageNet control (row 13 / rows 12–13): ${fmtPct(s.frozen_ctr_r13_pct)} / ${fmtPct(s.frozen_ctr_pct)}`;
        }
        showTooltip(ev, `
          <b>${s.label}</b> (${s.grid}×${s.grid}, ${s.rows_label}, unif ${fmtPct(s.uniform_pct)})<br>
          Fine-Tuned (${traj.objective_label}): <b class="ink-model">${fmtPct(s.ft_ctr_pct)}</b><br>
          Frozen ImageNet control: <b>${fmtPct(s.frozen_ctr_pct)}</b><br>
          Normal empty (${meta.n_other_test_empty} frames, ${meta.n_other_test_empty_cams} cams): <b>${fmtPct(s.ctrl_ctr_pct)}</b>
          <div class="tt-sep"></div>
          Standard ReLU Grad-CAM clipped: <b>${fmtPct(s.ft_std_clip_pct)}</b><br>
          Bar L2 energy (Fine-Tuned vs. ImageNet control): <b>${fmtPct(s.ft_l2_bar_pct)}</b> vs. ${fmtPct(s.frozen_l2_bar_pct)}<br>
          Global Linear CKA (Fine-Tuned vs. ImageNet control): <b>${s.cka_vs_frozen.toFixed(3)}</b>
          ${extra}
        `);
      });
      hit.addEventListener("mousemove", moveTooltip);
      hit.addEventListener("mouseleave", () => {
        guide.setAttribute("opacity", "0");
        hideTooltip();
      });
    });

    // Bottom Cohort Baseline Stacked Bar
    const baseCoh = data.interventions.baseline.cohorts[cohort];
    const cohNames = {
      all: `Camera 156 empty (${baseCoh.n})`,
      night: `Camera 156 Night IR (${baseCoh.n})`,
      day: `Camera 156 Day RGB (${baseCoh.n})`,
    };
    $("cohortBarTitle").textContent = cohNames[cohort];
    const frCoh = data.frozen_baseline.cohorts[cohort];
    const frRP = frCoh.rabbit_pct + frCoh.paca_pct;
    $("cohortBarMeta").textContent = `ImageNet control: ${fmtPct(frCoh.empty_pct, 1)} empty, ${fmtPct(frRP, 0)} rabbit/paca`;

    const bar = $("cohortStackedBar");
    bar.innerHTML = "";
    const segs = [
      { key: "rabbit", label: "Brazilian rabbit", short: "rabbit", pct: baseCoh.rabbit_pct, color: "#DDAA33" },
      { key: "paca", label: "Lowland paca", short: "paca", pct: baseCoh.paca_pct, color: "#9c6a00" },
      { key: "empty", label: "Empty", short: "empty", pct: baseCoh.empty_pct, color: "#004488" },
      { key: "other", label: "Other species", short: "other", pct: baseCoh.other_pct, color: "#cccccc" },
    ];
    segs.forEach((sg) => {
      const sp = document.createElement("span");
      sp.style.width = `${Math.max(0, sg.pct)}%`;
      sp.style.background = sg.color;
      sp.title = `${sg.label}: ${fmtPct(sg.pct)}`;
      bar.appendChild(sp);
    });

    $("cohortBreakdown").innerHTML = segs.map((sg) => `
      <span class="cb-item" title="${sg.label}: ${fmtPct(sg.pct)}">
        <span class="leg-swatch" style="background:${sg.color};${sg.key === "other" ? "border:1px solid #a8a8a8;" : ""}"></span>
        <b>${fmtPct(sg.pct)}</b> ${sg.short}
      </span>
    `).join("");
  }

  // =========================================================================
  // COLUMN 2: PHOTO + HEATMAP CANVAS + PIXEL-ALIGNED 18-STRIP OCCLUSION
  // =========================================================================
  function drawHeatmap(frame) {
    const canvas = $("heatCanvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 448, 448);

    const r12Guide = $("r12Guide");
    r12Guide.style.display = "none";

    const maps = frame.maps;
    const shortTarget = maps.target_common === "Brazilian rabbit"
      ? "rabbit"
      : (maps.target_common === "Lowland paca" ? "paca" : maps.target_common.toLowerCase());
    $("ctrPosLbl").textContent = `pushes ${shortTarget}`;

    const ctrLegend = $("ctrLegend");
    if (state.mode === "cover_bar") {
      ctrLegend.style.opacity = "0.45";
      $("mapNote").textContent = "Bottom 24 px covered on input";
      return;
    }

    ctrLegend.style.opacity = "1";
    const l49Active = state.mode === "zero_r13" || state.mode === "zero_r12_13" || state.mode === "zero_r6_7";
    const basePrefix = l49Active ? "Baseline map · " : "";
    drawDivergingGrid(ctx, maps.l49_contrastive, 14);
    r12Guide.style.display = "block";
    $("mapNote").textContent = `${basePrefix}Rows 12–13 hold ${fmtPct(maps.l49_ctr_r12_13_pct, 0)} of positive margin`;
  }

  function drawDivergingGrid(ctx, gridFloat, sz) {
    let maxPos = 1e-6;
    let maxNeg = 1e-6;
    for (let r = 0; r < sz; r++) {
      for (let c = 0; c < sz; c++) {
        const v = gridFloat[r][c];
        if (v > maxPos) maxPos = v;
        if (-v > maxNeg) maxNeg = -v;
      }
    }
    const off = document.createElement("canvas");
    off.width = sz;
    off.height = sz;
    const octx = off.getContext("2d");
    const imgData = octx.createImageData(sz, sz);
    const d = imgData.data;
    for (let r = 0; r < sz; r++) {
      for (let c = 0; c < sz; c++) {
        const raw = gridFloat[r][c];
        const idx = (r * sz + c) * 4;
        if (raw > 0) {
          const norm = Math.min(1, raw / maxPos);
          if (norm < 0.08) {
            d[idx + 3] = 0;
          } else {
            d[idx] = Math.round(238 - norm * 28);
            d[idx + 1] = Math.round(186 - norm * 56);
            d[idx + 2] = Math.round(48 - norm * 38);
            d[idx + 3] = Math.round(Math.pow(norm, 0.72) * 224);
          }
        } else {
          const norm = Math.min(1, -raw / maxNeg);
          if (norm < 0.12) {
            d[idx + 3] = 0;
          } else {
            d[idx] = 0;
            d[idx + 1] = 74;
            d[idx + 2] = 152;
            d[idx + 3] = Math.round(Math.pow(norm, 0.75) * 165);
          }
        }
      }
    }
    octx.putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, 448, 448);
  }

  function updateStripBand() {
    const band = $("stripBand");
    if (state.hoverStrip === null) {
      band.style.display = "none";
      return;
    }
    const y0 = 16 + state.hoverStrip * 24;
    band.style.display = "block";
    band.style.top = `${(y0 / 448) * 100}%`;
    band.style.height = `${(24 / 448) * 100}%`;
  }

  function renderStripChart() {
    const { data, cohort, mode } = state;
    const isCover = mode === "cover_bar";
    const strips = data.strips[cohort];
    const svg = $("stripChart");
    svg.innerHTML = "";

    // Dynamic viewBox width matching CSS aspect ratio -> 0px vertical offset vs 448x448 photo & 0% text stretch!
    const W = state.stripVbW;
    const H = 448;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("preserveAspectRatio", "none");

    // Wide unclipped range covering full dz0_black [-6.33, +9.39] across All, Night, and Day!
    const xMin = -7.0;
    const xMax = 10.5;
    const padL = 8;
    const padR = 8;
    const xAt = (v) => padL + ((Math.max(xMin, Math.min(xMax, v)) - xMin) / (xMax - xMin)) * (W - padL - padR);
    const x0 = xAt(0);

    // Top 0..16 band: crisp axis ticks matching unused y=0..16 above Strip 0
    for (const tk of [-6, -3, 0, 5, 10]) {
      const x = xAt(tk);
      const t = el("text", {
        x, y: 12, "text-anchor": "middle",
        "font-size": 10.5, fill: tk === 0 ? "#333" : "#888",
        "font-weight": tk === 0 ? "700" : "400",
      }, svg);
      t.textContent = tk > 0 ? `+${tk}` : (tk < 0 ? `−${Math.abs(tk)}` : "0");
    }

    // Vertical reference lines across y=16..448
    for (const tk of [-6, -3, 0, 5, 10]) {
      const x = xAt(tk);
      el("line", {
        x1: x, y1: 16, x2: x, y2: 448,
        stroke: tk === 0 ? "#999" : "#ececec",
        "stroke-width": tk === 0 ? 1.1 : 1,
      }, svg);
    }

    strips.forEach((st) => {
      const yMid = st.y0 + 12;
      const isBar = Boolean(st.is_bar);
      const isActive = isBar && isCover;

      if (isBar) {
        el("rect", {
          x: 0, y: st.y0, width: W, height: 24,
          fill: isActive ? "#e1eef7" : "#fbf3db",
        }, svg);
      }

      // Black-fill horizontal bar from x0 (unclipped!)
      const xb = xAt(st.dz0_black);
      el("rect", {
        x: Math.min(x0, xb),
        y: st.y0 + 5.5,
        width: Math.max(1.5, Math.abs(xb - x0)),
        height: 13,
        fill: isBar ? "#004488" : (st.dz0_black >= 0 ? "#6a8fb3" : "#c7ccd2"),
      }, svg);

      // Image-mean diamond marker
      const xm = xAt(st.dz0_mean);
      el("path", {
        d: `M${xm},${yMid - 4.8} L${xm + 4.8},${yMid} L${xm},${yMid + 4.8} L${xm - 4.8},${yMid} Z`,
        fill: isBar ? "#DDAA33" : "#806000",
        stroke: "white",
        "stroke-width": 1.0,
      }, svg);

      // Place Strip 17 value label cleanly left-aligned inside navy bar, well clear of the right-end gold diamond!
      if (isBar) {
        const lbl = el("text", {
          x: x0 + 5,
          y: st.y0 + 15.8,
          "text-anchor": "start",
          "font-size": 11,
          "font-weight": "700",
          fill: "white",
        }, svg);
        lbl.textContent = fmtSigned(st.dz0_black, 1);
      }

      const hit = el("rect", {
        x: 0, y: st.y0, width: W, height: 24,
        fill: "transparent",
        style: isBar ? "cursor:pointer;" : "cursor:default;",
      }, svg);
      hit.addEventListener("mouseenter", (ev) => {
        state.hoverStrip = st.strip;
        updateStripBand();
        hit.setAttribute("fill", "rgba(0, 68, 136, 0.12)");
        const extra = st.strip === 15
          ? `<div class="tt-sep"></div><i>Note: Strip 15 sits 24 px above the white bar inside Layer 22's receptive field — mean fill tips the thin margin (${fmtSigned(st.dz0_mean)}), whereas black fill lowers empty (${fmtSigned(st.dz0_black)}).</i>`
          : (isBar ? `<div class="tt-sep"></div><i>Click Strip 17 or "Cover bar" to toggle input bar occlusion.</i>` : "");
        showTooltip(ev, `
          <b>Strip ${st.strip}${isBar ? " (24 px white bar)" : ""}</b> · y = ${st.y0}–${st.y1} px<br>
          Black fill: <b>Δz_empty = ${fmtSigned(st.dz0_black)}</b> (${fmtPct(st.empty_pct_black)} empty)<br>
          Image-mean fill: <b>Δz_empty = ${fmtSigned(st.dz0_mean)}</b> (${fmtPct(st.empty_pct_mean)} empty)
          ${extra}
        `);
      });
      hit.addEventListener("mousemove", moveTooltip);
      hit.addEventListener("mouseleave", () => {
        state.hoverStrip = null;
        updateStripBand();
        hit.setAttribute("fill", "transparent");
        hideTooltip();
      });
      if (isBar) {
        hit.addEventListener("click", () => {
          setMode(state.mode === "cover_bar" ? "baseline" : "cover_bar");
        });
      }
    });
  }

  function renderCol2() {
    const { data, mode, frameIdx } = state;
    const frame = data.frames[frameIdx];
    const isCover = mode === "cover_bar";

    $("coverBarBtn").setAttribute("aria-pressed", String(isCover));

    const imgUrl = `/images/${encodeURIComponent(frame.filename)}?mode=model${isCover ? "&mask_bar=1" : ""}`;
    const imgEl = $("mainPhoto");
    if (imgEl.dataset.curUrl !== imgUrl) {
      imgEl.src = imgUrl;
      imgEl.dataset.curUrl = imgUrl;
    }
    $("photoBadge").textContent = `Camera 156 · ${frame.label}`;

    drawHeatmap(frame);

    const fState = frame.states[mode];
    const pill = $("framePredPill");
    pill.textContent = `${fState.pred_common} (${Math.round(fState.prob_pct)}%)`;
    pill.className = "pred-pill " + (fState.is_halluc ? "halluc" : (fState.is_empty ? "ok" : "neutral"));
    pill.title = `Logits — empty: ${fmtSigned(fState.z_empty)}, Brazilian rabbit: ${fmtSigned(fState.z_rabbit)}, lowland paca: ${fmtSigned(fState.z_paca)}`;
    $("frameGtText").textContent = `Label: ${frame.gt_common}`;
    $("frameGtText").title = `Ground truth label: ${fmtSpeciesLabel(frame.gt_common, frame.gt_sci)}`;

    renderStripChart();
  }

  // =========================================================================
  // COLUMN 3: EXACT 14-ROW ADDITIVE LOGIT PROFILE & SPATIALLY LOCAL FIX
  // =========================================================================
  function renderCol3() {
    const { data, cohort, mode } = state;
    const rows = data.layer49_rows[cohort];
    const intObj = data.interventions[mode];
    const cohData = intObj.cohorts[cohort];
    const ctrl = intObj.controls;
    const baseCoh = data.interventions.baseline.cohorts[cohort];
    const baseCtrl = data.interventions.baseline.controls;
    const meta = data.meta;

    $("intPills").querySelectorAll("button").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
    });
    $("coverBanner").classList.toggle("visible", mode === "cover_bar");
    const activeMeta = data.interventions_meta.find((m) => m.id === mode);
    $("col3ModeTag").textContent = activeMeta ? activeMeta.label : "";

    const zeroSet = new Set(activeMeta && Array.isArray(activeMeta.zero_rows) ? activeMeta.zero_rows : []);
    const clickableRows = new Set([6, 7, 12, 13]);

    const svg = $("rowChart");
    svg.innerHTML = "";
    const W = 390;
    const H = 264;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    const m = { t: 16, r: 44, b: 22, l: 48 };
    const pw = W - m.l - m.r;
    const ph = H - m.t - m.b;
    const rowH = ph / 14;

    const minV = -5.6;
    const maxV = 0.7;
    const xAt = (v) => m.l + ((Math.max(minV, Math.min(maxV, v)) - minV) / (maxV - minV)) * pw;
    const x0 = xAt(0);

    const axHdr = el("text", { x: xAt(-2.8), y: 11, "text-anchor": "middle", "font-size": 10, fill: "#666" }, svg);
    axHdr.textContent = "← suppresses empty (z₀ per row)";

    for (const tk of [-5, -4, -3, -2, -1, 0]) {
      const x = xAt(tk);
      el("line", {
        x1: x, y1: m.t, x2: x, y2: m.t + ph,
        stroke: tk === 0 ? "#999" : "#eeeeee",
        "stroke-width": tk === 0 ? 1.1 : 1,
      }, svg);
      const t = el("text", {
        x, y: m.t + ph + 14, "text-anchor": "middle",
        "font-size": 10, fill: "#666",
      }, svg);
      t.textContent = tk === 0 ? "0" : `−${Math.abs(tk)}`;
    }

    rows.forEach((r, idx) => {
      const y = m.t + idx * rowH;
      const isZeroed = zeroSet.has(r.row);
      const isHot = r.row >= 11;
      const isClickable = clickableRows.has(r.row);

      if (isZeroed) {
        el("rect", {
          x: m.l, y: y + 0.5, width: pw + 36, height: rowH - 1,
          fill: mode === "zero_r6_7" ? "#fbf3db" : "#e1eef7",
          opacity: 0.75,
        }, svg);
      }

      const rLbl = el("text", {
        x: m.l - 6, y: y + rowH * 0.72, "text-anchor": "end",
        "font-size": 10.5,
        "font-weight": (isHot || isZeroed) ? "700" : "400",
        fill: isZeroed ? (mode === "zero_r6_7" ? "#806000" : "#004488") : (isHot ? "#806000" : "#666"),
      }, svg);
      rLbl.textContent = `Row ${r.row}`;

      const xv = xAt(r.z_empty);
      const bx = Math.min(x0, xv);
      const bw = Math.max(1.5, Math.abs(xv - x0));
      el("rect", {
        x: bx, y: y + 2.5, width: bw, height: Math.max(4, rowH - 5),
        fill: isZeroed ? "none" : (isHot ? "#DDAA33" : "#5c7fa6"),
        stroke: isZeroed ? (mode === "zero_r6_7" ? "#806000" : "#004488") : (isHot ? "#806000" : "none"),
        "stroke-width": isZeroed ? 1.3 : (isHot ? 0.8 : 0),
        "stroke-dasharray": isZeroed ? "3,2" : null,
      }, svg);

      if (isZeroed) {
        const zt = el("text", {
          x: x0 + 6, y: y + rowH * 0.72, "font-size": 10, "font-weight": "700",
          fill: mode === "zero_r6_7" ? "#806000" : "#004488",
        }, svg);
        zt.textContent = "×0";
      } else if (isHot) {
        const vt = el("text", {
          x: bx - 4, y: y + rowH * 0.72, "text-anchor": "end",
          "font-size": 10, "font-weight": "700", fill: "#806000",
        }, svg);
        vt.textContent = fmtSigned(r.z_empty, 1);
      }

      const hit = el("rect", {
        x: 0, y, width: W, height: rowH, fill: "transparent",
        style: isClickable ? "cursor:pointer;" : "cursor:default;",
      }, svg);
      hit.addEventListener("mouseenter", (ev) => {
        hit.setAttribute("fill", "rgba(0, 68, 136, 0.08)");
        let clickTip = "";
        if (r.row === 13) clickTip = "<br><i>Click to toggle Zero row 13</i>";
        else if (r.row === 12) clickTip = "<br><i>Click to toggle Zero rows 12–13</i>";
        else if (r.row === 6 || r.row === 7) clickTip = "<br><i>Click to toggle Control rows 6–7</i>";
        showTooltip(ev, `
          <b>Layer 49 · Row ${r.row}</b> (nominal y = ${r.y0}–${r.y1} px)<br>
          Exact additive empty logit: <b class="${isHot ? "ink-model" : "ink-train"}">${fmtSigned(r.z_empty)}</b><br>
          Rabbit − empty margin: <b>${fmtSigned(r.margin_rabbit)}</b> · Paca − empty: <b>${fmtSigned(r.margin_paca)}</b><br>
          Activation norm ‖A₄‖: <b>${r.norm_c156.toFixed(1)}</b> (${r.norm_ratio.toFixed(2)}× normal empty)<br>
          Zeroing Row ${r.row} alone → <b>${fmtPct(r.zero_single_empty_pct)}</b> empty
          ${clickTip}
        `);
      });
      hit.addEventListener("mousemove", moveTooltip);
      hit.addEventListener("mouseleave", () => {
        hit.setAttribute("fill", "transparent");
        hideTooltip();
      });
      if (isClickable) {
        hit.addEventListener("click", () => {
          if (r.row === 13) setMode(state.mode === "zero_r13" ? "baseline" : "zero_r13");
          else if (r.row === 12) setMode(state.mode === "zero_r12_13" ? "baseline" : "zero_r12_13");
          else if (r.row === 6 || r.row === 7) setMode(state.mode === "zero_r6_7" ? "baseline" : "zero_r6_7");
        });
      }
    });

    // 2 Clean, Consistent Impact Metric Rows (both higher = better!)
    const c156Lbls = {
      all: `Camera 156 empty (${cohData.n})`,
      night: `Camera 156 Night IR empty (${cohData.n})`,
      day: `Camera 156 Day RGB empty (${cohData.n})`,
    };
    $("metricC156Lbl").textContent = c156Lbls[cohort];
    $("metricOtherLbl").textContent = `Other ${ctrl.other_ani_cams} test cameras (${ctrl.other_ani_n} animals)`;

    const m1Val = $("metricC156Val");
    m1Val.textContent = `${fmtPct(cohData.empty_pct)} empty`;
    m1Val.className = "metric-val " + (cohData.empty_pct >= 80 ? "ink-train" : "ink-model");
    $("metricC156Bar").style.width = `${cohData.empty_pct}%`;
    $("metricC156Bar").style.background = cohData.empty_pct >= 80 ? "#004488" : "#DDAA33";

    const dpp = ctrl.other_ani_delta_pp;
    const deltaStr = mode === "baseline" ? "" : ` (${dpp >= 0 ? "+" : "−"}${Math.abs(dpp).toFixed(1)} pp)`;
    const m3Val = $("metricOtherVal");
    m3Val.textContent = `${fmtPct(ctrl.other_ani_acc_pct)}${deltaStr}`;
    m3Val.className = "metric-val";
    $("metricOtherBar").style.width = `${ctrl.other_ani_acc_pct}%`;
    $("metricOtherBar").style.background = "#5c7fa6";

    const bindTip = (id, htmlFn) => {
      const node = $(id);
      node.onmouseenter = (ev) => showTooltip(ev, htmlFn());
      node.onmousemove = moveTooltip;
      node.onmouseleave = hideTooltip;
    };

    const isRowAblation = mode === "zero_r13" || mode === "zero_r12_13" || mode === "zero_r6_7";
    bindTip("metricC156", () => {
      const avgplNote = isRowAblation
        ? `<div class="tt-sep"></div>Exact avgpool parity: <b>${fmtPct(cohData.avgpl_empty_pct)}</b> empty (z_empty = ${fmtSigned(cohData.avgpl_z_empty)}) — identical argmax because max |fc bias| = ${meta.fc_bias_max_abs.toFixed(3)}.`
        : "";
      return `
        <b>${c156Lbls[cohort]} · ${activeMeta.label}</b><br>
        Empty: <b class="ink-train">${fmtPct(cohData.empty_pct)}</b> (baseline ${fmtPct(baseCoh.empty_pct)})<br>
        Brazilian rabbit: <b>${fmtPct(cohData.rabbit_pct)}</b> · Lowland paca: <b>${fmtPct(cohData.paca_pct)}</b><br>
        Mean logits: z_empty = <b>${fmtSigned(cohData.z_empty)}</b>, z_rabbit = ${fmtSigned(cohData.z_rabbit)}, z_paca = ${fmtSigned(cohData.z_paca)}
        ${avgplNote}
      `;
    });

    bindTip("metricOther", () => `
      <b>Unseen test control (${ctrl.other_ani_n} real animals · ${ctrl.other_ani_cams} cameras)</b><br>
      Top-1 species accuracy: <b>${fmtPct(ctrl.other_ani_acc_pct)}</b> (${dpp >= 0 ? "+" : "−"}${Math.abs(dpp).toFixed(1)} pp vs baseline)<br>
      Frames lost / gained vs baseline: <b>${ctrl.other_ani_lost}</b> lost, <b>${ctrl.other_ani_gained}</b> gained<br>
      Other ${meta.n_other_test_empty} empty frames (${meta.n_other_test_empty_cams} cams): <b>${fmtPct(ctrl.other_emp_acc_pct)}</b> empty (baseline ${fmtPct(baseCtrl.other_emp_acc_pct)})<br>
      Camera 156 real animals (${ctrl.c156_ani_n}): <b>${ctrl.c156_ani_rab_paca}/${ctrl.c156_ani_n}</b> rabbit/paca hallucinations (baseline ${baseCtrl.c156_ani_rab_paca}/${baseCtrl.c156_ani_n})
    `);
  }

  // =========================================================================
  // BOTTOM STRIP: FULL 129,809-TRAIN NEAREST NEIGHBORS (LIVE WITH COL 2 & 3!)
  // =========================================================================
  function renderBottomStrip() {
    const { data, cohort, mode, knnBackbone, knnScope, frameIdx } = state;
    const isFrozen = knnBackbone === "frozen";
    const isPredScope = knnScope === "pred";
    const frame = data.frames[frameIdx];
    const meta = data.meta;

    $("knnBackboneSeg").querySelectorAll("button").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.bb === knnBackbone));
    });
    $("knnScopeSeg").querySelectorAll("button").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.scope === knnScope));
    });

    const cohObj = isFrozen
      ? data.frozen_baseline.cohorts[cohort]
      : data.interventions[mode].cohorts[cohort];
    const knn = isPredScope ? cohObj.knn_pred : cohObj.knn;
    const knnAll = cohObj.knn;

    const activeMeta = data.interventions_meta.find((m) => m.id === mode);
    const modeLabel = isFrozen ? "Frozen ImageNet control (unmodified)" : activeMeta.label;
    const scopeLabel = isPredScope ? "Predicted class (ExMatchina)" : "All classes (unfiltered)";
    $("knnSubline").textContent = `${fmtInt(meta.n_train_embeddings)} train · ${isFrozen ? "ImageNet control" : activeMeta.label}`;

    const cohTag = cohort === "all" ? `All ${cohObj.n}` : (cohort === "night" ? `Night ${cohObj.n}` : `Day ${cohObj.n}`);
    $("knnVoteLabel").textContent = `${cohTag} · top-5 votes (${fmtInt(knn.n_votes)})`;
    $("knnCosLabel").textContent = `cos ${knn.top1_cos.toFixed(3)}`;

    // Stacked camera-group vote bar + dynamic legend covering EVERY visible segment!
    const bar = $("knnStackedBar");
    bar.innerHTML = "";
    const parts = [
      { pct: knn.sa_wb_pct, color: "#DDAA33", label: "White-bar S. America" },
      { pct: knn.cam0_pct, color: "#eed99b", label: "Camera 0 (white bar)" },
      { pct: knn.cam186_pct, color: "#8a99a8", label: "Camera 186" },
      { pct: knn.normal_empty_pct, color: "#004488", label: "Normal empty" },
      { pct: knn.other_nonempty_pct, color: "#cfd5dc", label: "Other" },
    ];
    parts.forEach((p) => {
      if (p.pct < 0.05) return;
      const sp = document.createElement("span");
      sp.style.width = `${p.pct}%`;
      sp.style.background = p.color;
      bar.appendChild(sp);
    });

    // Always include White-bar S. America + Normal empty, plus every other segment >= 0.5%
    const visibleLegend = parts.filter((p) =>
      p.label === "White-bar S. America" || p.label === "Normal empty" || p.pct >= 0.5
    );
    $("knnLegend").innerHTML = visibleLegend
      .map((it) => `<span class="leg-item"><span class="leg-swatch" style="background:${it.color}"></span>${it.label}: ${fmtPct(it.pct, 1)}</span>`)
      .join("");

    const fStateKey = isFrozen ? "frozen" : mode;
    const activePred = frame.states[fStateKey].pred_common;
    $("knnFootNote").textContent = isFrozen
      ? "ImageNet control only (Col 2–3 edits apply to Fine-Tuned)"
      : (isPredScope
          ? `Tiles: nearest ${activePred} frames · click photo to zoom`
          : `Tiles: ${frame.label} (all classes) · click to zoom`);

    const vb = $("knnVoteBlock");
    vb.onmouseenter = (ev) => {
      const topRows = knn.top_cameras
        .map((c) => `${c.camera_name}: <b>${fmtPct(c.share_pct, 1)}</b> (${fmtPct(c.empty_pct, 0)} empty${c.has_white_bar ? ", white bar" : ""})`)
        .join("<br>");
      const unfilteredNote = isPredScope
        ? `<br>Unfiltered all-class White-bar S. America: <b>${fmtPct(knnAll.sa_wb_pct, 1)}</b> (normal empty: <b>${fmtPct(knnAll.normal_empty_pct, 1)}</b>)`
        : "";
      showTooltip(ev, `
        <b>${fmtInt(meta.n_train_embeddings)}-train cosine kNN · ${modeLabel}</b><br>
        Filter: <b>${scopeLabel}</b> (${cohTag}, ${fmtInt(knn.n_votes)} top-5 votes)<br>
        Total empty votes: <b>${fmtPct(knn.total_empty_pct, 1)}</b> (normal non-bar: <b>${fmtPct(knn.normal_empty_pct, 1)}</b>)${unfilteredNote}
        <div class="tt-sep"></div>
        <b>Top neighbor cameras:</b><br>${topRows}
      `);
    };
    vb.onmousemove = moveTooltip;
    vb.onmouseleave = hideTooltip;

    // 5 Neighbor Photo Tiles for selected Camera 156 frame
    // Use native 4:3 URL (cached 1 hr by server) -> minimal pillarboxing & full bottom white bar visible!
    const neighbors = isPredScope
      ? frame.states[fStateKey].neighbors_pred
      : frame.states[fStateKey].neighbors;
    const container = $("neighborTiles");
    container.innerHTML = "";

    neighbors.forEach((nb, rank) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ntile";

      const trackCls = nb.is_sa_wb
        ? "sa-wb"
        : (nb.is_cam0
            ? "cam0-wb"
            : ((nb.is_empty && !nb.has_white_bar) ? "norm-emp" : "other-cam"));
      const imgUrl = `/images/${encodeURIComponent(nb.filename)}`;

      btn.innerHTML = `
        <div class="ntile-photo">
          <img src="${imgUrl}" alt="${nb.camera_name} · ${nb.species_common}">
          <div class="animal-bbox"></div>
          <span class="ntile-rank">#${rank + 1}</span>
        </div>
        <div class="ntile-track ${trackCls}">
          <span>${nb.camera_name} · ${nb.species_common}</span>
        </div>
      `;

      const imgEl = btn.querySelector(".ntile-photo img");
      const boxEl = btn.querySelector(".animal-bbox");
      if (nb.bbox) {
        boxEl._bbox = nb.bbox;
        boxEl._img = imgEl;
        imgEl.addEventListener("load", () => placeBbox(boxEl, imgEl, nb.bbox));
        if (imgEl.complete && imgEl.naturalWidth) {
          requestAnimationFrame(() => placeBbox(boxEl, imgEl, nb.bbox));
        }
      }

      const spFull = fmtSpeciesLabel(nb.species_common, nb.species_sci);
      btn.addEventListener("mouseenter", (ev) => {
        showTooltip(ev, `
          <b>#${rank + 1} Nearest train frame · ${nb.camera_name}</b><br>
          Label: <b>${spFull}</b><br>
          Cosine similarity: <b>${nb.cos_sim.toFixed(3)}</b><br>
          ${nb.camera_name} training empty rate: <b>${fmtPct(nb.cam_empty_pct, 1)}</b><br>
          Bottom 24 px white bar: <b>${nb.has_white_bar ? "Yes (matches Camera 156)" : "No"}</b>
          <div class="tt-sep"></div>
          <i>Click photo to zoom full resolution</i>
        `);
      });
      btn.addEventListener("mousemove", moveTooltip);
      btn.addEventListener("mouseleave", hideTooltip);
      btn.addEventListener("click", () => {
        hideTooltip();
        openZoom(
          `#${rank + 1} Neighbor · ${nb.camera_name} (${nb.species_common})`,
          imgUrl,
          `${nb.camera_name} (train split) · Label: ${spFull} · Cosine similarity: ${nb.cos_sim.toFixed(3)} · ${nb.camera_name} empty rate: ${fmtPct(nb.cam_empty_pct, 1)} · ${nb.has_white_bar ? "Has 24 px bottom white bar" : "No white bar"}.`,
          nb.bbox || null
        );
      });
      container.appendChild(btn);
    });
  }

  // =========================================================================
  // STATE CONTROLLERS & EVENT BINDING
  // =========================================================================
  function setCohort(next) {
    if (state.cohort === next) return;
    const prev = state.cohort;
    if (prev === "all") state.prevAllFrameIdx = state.frameIdx;
    state.cohort = next;
    $("cohortSeg").querySelectorAll("button").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.cohort === next));
    });

    if (next === "all") {
      state.frameIdx = state.prevAllFrameIdx;
      $("frameSelect").value = String(state.frameIdx);
    } else {
      const cur = state.data.frames[state.frameIdx];
      if (cur.lighting !== next) {
        const matchIdx = state.data.frames.findIndex((f) => f.lighting === next && !f.is_real_animal);
        if (matchIdx >= 0) {
          state.frameIdx = matchIdx;
          $("frameSelect").value = String(matchIdx);
        }
      }
    }
    renderAll();
  }

  function setMode(next) {
    state.mode = next;
    state.knnBackbone = "ft";
    renderAll();
  }

  function renderAll() {
    renderCol1();
    renderCol2();
    renderCol3();
    renderBottomStrip();
  }

  function wireControls() {
    const meta = state.data.meta;
    $("topbarSub").textContent = `White-box diagnosis on ${fmtInt(meta.n_empty_all)} empty test frames · Fine-Tuned vs. Frozen ImageNet control`;
    $("btnCohAll").textContent = `All (${meta.n_empty_all})`;
    $("btnCohNight").textContent = `Night IR (${meta.n_empty_night})`;
    $("btnCohDay").textContent = `Day RGB (${meta.n_empty_day})`;

    $("cohortSeg").addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-cohort]");
      if (btn) setCohort(btn.dataset.cohort);
    });

    $("coverBarBtn").addEventListener("click", () => {
      setMode(state.mode === "cover_bar" ? "baseline" : "cover_bar");
    });

    $("clearCoverBtn").addEventListener("click", () => {
      setMode("baseline");
    });

    $("intPills").addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-mode]");
      if (btn) setMode(btn.dataset.mode);
    });

    $("knnBackboneSeg").addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-bb]");
      if (!btn) return;
      state.knnBackbone = btn.dataset.bb;
      renderBottomStrip();
    });

    $("knnScopeSeg").addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-scope]");
      if (!btn) return;
      state.knnScope = btn.dataset.scope;
      renderBottomStrip();
    });

    const sel = $("frameSelect");
    state.data.frames.forEach((fr, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = fr.label;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => {
      state.frameIdx = Number(sel.value);
      if (state.cohort === "all") state.prevAllFrameIdx = state.frameIdx;
      renderCol2();
      renderBottomStrip();
    });

    const openMainZoom = () => {
      const fr = state.data.frames[state.frameIdx];
      const fState = fr.states[state.mode];
      const isCover = state.mode === "cover_bar";
      const url = isCover
        ? `/images/${encodeURIComponent(fr.filename)}?mode=model&mask_bar=1`
        : `/images/${encodeURIComponent(fr.filename)}`;
      const activeMeta = state.data.interventions_meta.find((m) => m.id === state.mode);
      openZoom(
        `Camera 156 · ${fr.label} (${activeMeta.label})`,
        url,
        `Ground truth: ${fmtSpeciesLabel(fr.gt_common, fr.gt_sci)} · Active prediction (${activeMeta.label}): ${fState.pred_common} (${Math.round(fState.prob_pct)}%) · Logits — empty: ${fmtSigned(fState.z_empty)}, Brazilian rabbit: ${fmtSigned(fState.z_rabbit)}, lowland paca: ${fmtSigned(fState.z_paca)}.`
      );
    };
    $("photoBox").addEventListener("click", openMainZoom);
    $("photoBox").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        openMainZoom();
      }
    });

    // Keep photo exact 1:1 square, strip chart 0px-aligned, and animal bboxes locked on resize
    const syncAllBboxes = () => {
      document.querySelectorAll("#neighborTiles .animal-bbox").forEach((b) => {
        if (b._bbox && b._img) placeBbox(b, b._img, b._bbox);
      });
      if (imageDialog.open && activeDialogBbox) {
        placeBbox($("dialogBbox"), $("dialogImg"), activeDialogBbox);
      }
    };
    const ro = new ResizeObserver(() => {
      syncSpatialSquare();
      syncAllBboxes();
    });
    ro.observe($("spatialPair"));
    ro.observe($("neighborTiles"));
    window.addEventListener("resize", syncAllBboxes);
    syncSpatialSquare();
  }

  async function init() {
    try {
      const resp = await fetch(DATA_URL, { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      state.data = await resp.json();
      wireControls();
      renderAll();
      syncSpatialSquare();
    } catch (err) {
      document.body.innerHTML = `<div style="padding:24px;font-family:sans-serif">Failed to load ${DATA_URL}: ${err}</div>`;
    }
  }

  init();
})();
