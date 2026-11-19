// Scrollama wrapper: one instance per [data-scrolly] block; marks the active step and calls onStep.
// Steps may sit inside closed <details class="more"> boxes: they take no space until opened, and closing a box
// whose step is showing falls back to the last visible step above it.

export function setupScrolly(name, onStep, { offset = 0.55 } = {}) {
  const root = document.querySelector(`[data-scrolly="${name}"]`);
  const steps = Array.from(root.querySelectorAll(".step"));
  const scroller = scrollama();
  let active = steps[0];

  const activate = (element, index, direction) => {
    active = element;
    steps.forEach((s) => s.classList.toggle("is-active", s === element));
    onStep(index, element, direction);
  };

  scroller
    .setup({ step: steps, offset })
    .onStepEnter(({ element, index, direction }) => activate(element, index, direction));

  // scrolled back above the first step: keep showing the first state
  scroller.onStepExit(({ index, direction }) => {
    if (index === 0 && direction === "up") activate(steps[0], 0, "up");
  });

  // "toggle" does not bubble: listen in the capture phase
  root.addEventListener("toggle", (event) => {
    scroller.resize();
    const box = event.target;
    if (box.open || !box.contains(active)) return;
    const shown = steps.filter((s) => !s.closest("details:not([open])"));
    const before = shown.filter((s) => s.compareDocumentPosition(box) & Node.DOCUMENT_POSITION_FOLLOWING);
    const back = before.at(-1) || steps[0];
    activate(back, steps.indexOf(back), "up");
  }, true);

  activate(steps[0], 0, "down");
  window.addEventListener("resize", () => scroller.resize());
  return scroller;
}
