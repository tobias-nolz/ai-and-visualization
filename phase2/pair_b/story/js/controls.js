// Small controls next to a chart: dropdown, toggle buttons, slider, checkbox. Each sits in the chart's
// <div class="controls"> and calls back with the new value; the chart redraws itself.
//
// Presenting stays keyboard-driven (js/keys.js): after a mouse or touch interaction a control hands the focus back,
// so arrow keys and space keep jumping between beats instead of moving the slider or pressing the button again.
// Reached with tab, controls keep the focus and work with the keyboard as usual.

function releaseAfterPointer(el, events, source = el) {
  let pointer = false;
  source.addEventListener("pointerdown", () => { pointer = true; });
  el.addEventListener("blur", () => { pointer = false; });
  for (const type of events) {
    el.addEventListener(type, () => {
      if (!pointer) return;
      pointer = false;
      el.blur();
    });
  }
}

function group(parent, label) {
  const g = parent.append("div").attr("class", "ctl");
  if (label) g.append("span").attr("class", "ctl-label").text(label);
  return g;
}

/** Dropdown. options: [{ value, label }]. */
export function dropdown(parent, { label, options, value, onChange }) {
  const g = group(parent, null);
  const id = `ctl-${Math.random().toString(36).slice(2, 9)}`;
  if (label) g.append("label").attr("class", "ctl-label").attr("for", id).text(label);
  const sel = g.append("select").attr("id", id);
  sel.selectAll("option").data(options).join("option").attr("value", (d) => d.value).text((d) => d.label);
  sel.property("value", value);
  sel.on("change", () => onChange(sel.property("value")));
  releaseAfterPointer(sel.node(), ["change"]);
  return {
    set(v) { sel.property("value", v); },
    node: g,
  };
}

/** Toggle buttons, one pressed. options: [{ value, label, title? }]. */
export function toggle(parent, { label, options, value, onChange, ariaLabel }) {
  const g = group(parent, label);
  const seg = g.append("div").attr("class", "seg").attr("role", "group").attr("aria-label", ariaLabel || label);
  const buttons = seg.selectAll("button").data(options).join("button").attr("type", "button")
    .attr("title", (d) => d.title || null)
    .text((d) => d.label)
    .on("click", (_, d) => {
      if (d.value === current) return;
      set(d.value);
      onChange(d.value);
    });
  buttons.each(function () { releaseAfterPointer(this, ["click"]); });
  let current = value;
  function set(v) {
    current = v;
    buttons.attr("aria-pressed", (d) => String(d.value === v));
  }
  set(value);
  return {
    set,
    /** Disable some options (they stay visible), e.g. picks that do not exist. */
    enable(isEnabled) {
      buttons.property("disabled", (d) => !isEnabled(d.value));
    },
    node: g,
  };
}

/** Slider over a list of values (steps need not be even). format(v) gives the text next to it. */
export function slider(parent, { label, values, value, format = String, onInput, width = 120 }) {
  const g = group(parent, null);
  const id = `ctl-${Math.random().toString(36).slice(2, 9)}`;
  if (label) g.append("label").attr("class", "ctl-label").attr("for", id).text(label);
  const input = g.append("input").attr("type", "range").attr("id", id)
    .attr("min", 0).attr("max", values.length - 1).attr("step", 1)
    .style("width", `${width}px`);
  const out = g.append("output").attr("for", id).attr("class", "ctl-value");
  function set(v) {
    const i = values.indexOf(v);
    input.property("value", i);
    out.text(format(v));
    input.attr("aria-valuetext", format(v));
  }
  input.on("input", () => {
    const v = values[+input.property("value")];
    out.text(format(v));
    input.attr("aria-valuetext", format(v));
    onInput(v);
  });
  releaseAfterPointer(input.node(), ["change", "pointerup"]);
  set(value);
  return { set, node: g };
}

/** Checkbox. */
export function checkbox(parent, { label, checked, onChange }) {
  const g = parent.append("label").attr("class", "ctl ctl-check");
  const input = g.append("input").attr("type", "checkbox").property("checked", checked);
  g.append("span").text(label);
  input.on("change", () => onChange(input.property("checked")));
  releaseAfterPointer(input.node(), ["change"], g.node()); // a click on the text toggles it too
  return { set(v) { input.property("checked", v); }, node: g };
}
