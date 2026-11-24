import { jpeg, request } from "./api.js";
import { el, notice, score } from "./ui.js";

export function validateReferencePreview(data, item) {
  if (typeof item.pending_id !== "string" || !item.pending_id || !jpeg(data.preview) || data.pending_id !== item.pending_id || data.identity !== item.identity
      || data.filename !== item.filename || data.faces !== item.faces
      || !Number.isInteger(data.width) || data.width < 1 || !Number.isInteger(data.height) || data.height < 1
      || !Array.isArray(data.candidates) || data.candidates.length !== item.faces
      || !data.candidates.every((face, i) => face?.index === i && Number.isFinite(face.confidence)
        && Array.isArray(face.bbox) && face.bbox.length === 4 && face.bbox.every(Number.isFinite)
        && (face.crop === null || jpeg(face.crop)))) {
    throw new Error("The reference preview changed. Refresh gallery status.");
  }
  return data;
}

export const candidateLabel = index => `Face ${index + 1}`;

export function createReferenceReview(item, submit, close) {
  const automatic = item.automatic_selection;
  const layout = el("div", null, "pending-body");
  const previewArea = el("div", null, "pending-preview");
  const message = el("div", null, "notice");
  message.setAttribute("role", "status");
  const retry = el("button", "Retry preview");
  const viewer = el("div", null, "pending-viewer");
  viewer.tabIndex = 0;
  viewer.setAttribute("aria-label", "Scrollable reference photograph");
  const stage = el("div", null, "pending-stage");
  const photo = el("img");
  photo.alt = `Reference photograph for ${item.identity}`;
  stage.append(photo);
  viewer.append(stage);

  const tools = el("div", null, "toolbar wrap pending-zoom");
  const fit = el("button", "Fit");
  const zoomOut = el("button", "−");
  const zoomIn = el("button", "+");
  const reset = el("button", "Reset zoom");
  zoomOut.setAttribute("aria-label", "Zoom out reference");
  zoomIn.setAttribute("aria-label", "Zoom in reference");
  const zoomLabel = el("output");
  tools.append(fit, zoomOut, zoomLabel, zoomIn, reset);
  previewArea.append(message, retry, viewer, tools);

  const options = el("div");
  options.append(el("h3", item.identity), el("p", `${item.filename} · ${item.faces} detected faces`));
  if (automatic) {
    options.append(el("p", `${candidateLabel(automatic.selected_face_index)} was automatically selected because its box was ${automatic.area_ratio.toFixed(2)} times the area of the next-largest box and it had the highest detection score (${score(automatic.detection_score)}).`));
  }
  const fields = el("fieldset");
  fields.append(el("legend", "Choose the matching face"));
  const cards = el("div", null, "pending-cards");
  fields.append(cards);
  const controls = el("div", null, "toolbar wrap");
  const confirm = el("button", automatic ? "Save selection" : "Confirm selection", "primary");
  const skip = el("button", automatic ? "Close review" : "Skip reference");
  controls.append(confirm, skip);
  options.append(fields, controls);
  layout.append(previewArea, options);

  let controller, preview, selected, busy = false;
  let zoom = 1;
  let choices = [];
  function updateControls() {
    fields.disabled = busy || !preview;
    confirm.disabled = fields.disabled || selected == null;
    skip.disabled = busy;
    for (const choice of choices) choice.box.disabled = busy;
  }

  function select(index, bringIntoView) {
    if (busy || !preview) return;
    selected = index;
    for (const choice of choices) {
      const active = choice.index === index;
      choice.radio.checked = active;
      choice.box.setAttribute("aria-pressed", String(active));
      if (active && bringIntoView && !choice.box.hidden) {
        const box = choice.box.getBoundingClientRect();
        const viewport = viewer.getBoundingClientRect();
        if (box.left < viewport.left || box.right > viewport.right
            || box.top < viewport.top || box.bottom > viewport.bottom) {
          viewer.scrollBy(box.left - viewport.left + (box.width - viewport.width) / 2,
                          box.top - viewport.top + (box.height - viewport.height) / 2);
        }
      }
    }
    updateControls();
  }

  function applyZoom() {
    if (!preview || !viewer.clientWidth || !viewer.clientHeight) return;
    const scale = Math.min(viewer.clientWidth / preview.width, viewer.clientHeight / preview.height);
    stage.style.width = `${Math.max(1, Math.floor(preview.width * scale * zoom))}px`;
    stage.style.height = `${Math.max(1, Math.floor(preview.height * scale * zoom))}px`;
    zoomLabel.textContent = `${Math.round(zoom * 100)}% of fit`;
    zoomOut.disabled = zoom <= 1;
    zoomIn.disabled = zoom >= 4;
  }

  function setZoom(value) {
    zoom = Math.min(4, Math.max(1, value));
    applyZoom();
    if (zoom === 1) viewer.scrollTo(0, 0);
    else if (selected != null) select(selected, true);
  }

  const observer = new ResizeObserver(applyZoom);
  observer.observe(viewer);
  fit.addEventListener("click", () => setZoom(1));
  reset.addEventListener("click", () => setZoom(1));
  zoomOut.addEventListener("click", () => setZoom(zoom / 1.25));
  zoomIn.addEventListener("click", () => setZoom(zoom * 1.25));

  function renderCandidate(face) {
    const label = candidateLabel(face.index);
    const card = el("label", null, "pending-candidate");
    const radio = el("input");
    radio.type = "radio";
    radio.name = automatic ? "automatic-face" : "pending-face";
    radio.value = String(face.index);
    radio.addEventListener("change", () => select(face.index, true));
    const info = el("span", null, "pending-candidate-info");
    info.append(el("strong", label), el("small", `Detection score ${score(face.confidence)}`));
    const portrait = el("span", null, "pending-candidate-crop");
    if (face.crop) {
      const image = el("img");
      image.src = face.crop;
      image.alt = "";
      portrait.append(image);
    } else {
      portrait.textContent = "Crop unavailable";
    }
    card.append(portrait, radio, info);
    cards.append(card);

    const box = el("button", null, "pending-face-box");
    box.append(el("span", String(face.index + 1)));
    box.type = "button";
    box.setAttribute("aria-label", label);
    box.setAttribute("aria-pressed", "false");
    const [x, y, width, height] = face.bbox;
    box.hidden = width <= 0 || height <= 0;
    box.style.left = `${100 * x / preview.width}%`;
    box.style.top = `${100 * y / preview.height}%`;
    box.style.width = `${100 * width / preview.width}%`;
    box.style.height = `${100 * height / preview.height}%`;
    box.addEventListener("click", () => select(face.index, false));
    stage.append(box);
    return { index: face.index, radio, box };
  }

  async function load() {
    const current = new AbortController();
    controller = current;
    retry.hidden = viewer.hidden = tools.hidden = true;
    notice(message, "Loading numbered reference preview…");
    try {
      const path = automatic ? `/gallery/automatic/${item.revision}/${item.pending_id}`
        : `/gallery/pending/${item.pending_id}?interactive=true`;
      const data = await request(path, { signal: current.signal });
      if (current.signal.aborted) return;
      preview = validateReferencePreview(data, item);
      photo.src = preview.preview;
      choices = preview.candidates.map(renderCandidate);
      if (automatic) select(item.selected_face_index, false);
      viewer.hidden = tools.hidden = false;
      applyZoom();
      notice(message);
    } catch (error) {
      if (current.signal.aborted) return;
      notice(message, error.message, "error");
      retry.hidden = false;
    }
    updateControls();
  }

  confirm.addEventListener("click", () => { if (!confirm.disabled) submit(item, selected); });
  skip.addEventListener("click", () => {
    if (skip.disabled) return;
    if (automatic) close();
    else submit(item);
  });
  retry.addEventListener("click", load);
  updateControls();
  load();
  return {
    element: layout,
    setBusy(value) { busy = value; updateControls(); },
    dispose() { controller?.abort(); observer.disconnect(); },
  };
}
