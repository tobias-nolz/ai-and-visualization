// Intro animations, as in the slide clips: a chart builds up once when it first scrolls into view, and a "Replay"
// button runs it again. With reduced motion (or without IntersectionObserver) charts are simply drawn finished.
//
// The chart's draw function reads anim.mode:
//   "start"  before the intro: the empty first frame (so the chart does not flash finished, then reset)
//   "play"   draw with d3 transitions from the first frame to the finished chart
//   "end"    finished chart, no transitions (resizes, control changes)

const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");

/** Delay + duration helper: a transition on selection `sel` that starts `delay` ms after the intro starts. */
export function tr(sel, delay = 0, duration = 600, ease = d3.easeCubicOut) {
  return sel.transition().delay(delay).duration(duration).ease(ease);
}

/**
 * fig: the <figure> (d3 selection); redraw(): draws the chart (may return a promise, e.g. while a photo loads);
 * target: the element that has to scroll into view (default: the figure's .chart-body).
 */
export function introAnimation(fig, redraw, { target = ".chart-body", threshold = 0.35 } = {}) {
  const canPlay = !reduce.matches && "IntersectionObserver" in window;
  const anim = { mode: canPlay ? "start" : "end" };

  anim.play = async () => {
    if (reduce.matches) return;
    anim.mode = "play";
    try { await redraw(); } finally { anim.mode = "end"; }
  };
  // a control was used before the intro ran: show the chart finished
  anim.skip = () => { if (anim.mode === "start") anim.mode = "end"; };

  if (canPlay) {
    const io = new IntersectionObserver((entries) => {
      if (!entries.some((e) => e.isIntersecting)) return;
      io.disconnect();
      if (anim.mode === "start") anim.play();
    }, { threshold, rootMargin: "0px 0px -10% 0px" });
    io.observe(fig.select(target).node());
  }

  if (!reduce.matches) {
    fig.select(".fig-head").classed("has-replay", true).insert("button", ":first-child")
      .attr("type", "button").attr("class", "replay").attr("title", "Play the chart's build-up again")
      .text("↻ Replay")
      .on("click", function () { this.blur(); anim.play(); });
  }
  return anim;
}
