export const $ = id => document.getElementById(id);

export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

export function icon(name) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  const use = document.createElementNS(ns, "use");
  svg.setAttribute("aria-hidden", "true");
  use.setAttribute("href", "#" + name);
  svg.append(use);
  return svg;
}

export const ms = value => Number.isFinite(value) ? `${value.toFixed(1)} ms` : "Unavailable";
export const score = value => Number.isFinite(value) ? value.toFixed(3) : "Unavailable";

export function notice(target, message = "", tone = "") {
  target.hidden = !message;
  target.dataset.tone = tone;
  target.textContent = message;
}

export function setStatus(target, message, tone) {
  target.className = `status ${tone}`;
  target.textContent = message;
}

export function details(target, entries) {
  target.replaceChildren(...entries.flatMap(([label, value]) => [el("dt", label), el("dd", value ?? "Unavailable")]));
}

export function tiles(target, entries) {
  target.replaceChildren(...entries.map(([label, value, tone]) => {
    const tile = el("div");
    if (tone) tile.dataset.tone = tone;
    tile.append(el("strong", value), el("small", label));
    return tile;
  }));
}

export function busyNotice(target, label, started) {
  const progress = el("progress");
  progress.setAttribute("aria-label", label);
  const timer = el("span");
  timer.setAttribute("aria-hidden", "true");
  const update = () => { timer.textContent = `${Math.floor((performance.now() - started) / 1000)} s elapsed`; };
  notice(target, label);
  target.prepend(progress);
  target.append(timer);
  update();
  const interval = setInterval(update, 250);
  return () => clearInterval(interval);
}
