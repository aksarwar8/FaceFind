import { jpeg } from "./api.js";
import { exportAnalysis } from "./exports.js";
import { $, el, ms, score, tiles } from "./ui.js";

const statuses = {
  recognised: ["Recognized", "green"],
  possible: ["Possible match", "amber"],
  unknown: ["Unknown", ""],
  error: ["Processing error", "red"],
};
const CROP_SIZE = 80;

export function validateAnalysis(data, status) {
  if (!jpeg(data.annotated_image) || data.pipeline_id !== status.active_pipeline_id
      || data.gallery_revision !== status.gallery_revision || !Number.isInteger(data.face_count)
      || !Number.isInteger(data.image_width) || data.image_width < 1
      || !Number.isInteger(data.image_height) || data.image_height < 1
      || !Array.isArray(data.results) || data.results.length !== data.face_count
      || !data.results.every(face => face && Object.hasOwn(statuses, face.status)
        && Number.isInteger(face.face_number))
      || !["recognised", "possible", "unknown", "errors"].every(key => Number.isInteger(data.summary?.[key]))) {
    throw new Error("The backend returned an invalid or incompatible analysis.");
  }
  return data;
}

export function summaryEntries(result) {
  const { recognised, unknown, possible, errors } = result.summary;
  const entries = [["Detected", result.face_count, "blue"], ["Recognized", recognised, "green"], ["Unknown", unknown]];
  if (result.pipeline_id === "yunet_sface" || possible) entries.push(["Possible matches", possible, "amber"]);
  entries.push(["Processing errors", errors, errors ? "red" : null]);
  return entries;
}

function validBox(box) {
  return Array.isArray(box) && box.length === 4 && box.every(Number.isFinite) && box[2] > 0 && box[3] > 0;
}

function resultCard(face, index, select) {
  const card = el("article", null, "face-card");
  const [status, tone] = statuses[face.status];
  const name = face.status === "recognised" ? face.identity : status;
  const button = el("button", null, "face-row");
  button.dataset.index = index;
  button.setAttribute("aria-pressed", "false");
  button.disabled = !validBox(face.bbox);
  if (button.disabled) button.title = "Face position is unavailable";
  button.addEventListener("click", () => select(index));
  const crop = el("canvas", null, "face-crop");
  crop.width = crop.height = CROP_SIZE;
  crop.setAttribute("aria-hidden", "true");
  const body = el("span", null, "face-card-info");
  body.append(el("small", `Face ${face.face_number}`, "face-number"), el("strong", name),
    el("small", `Cosine similarity ${score(face.best_score)}`), el("span", status, `tag ${tone}`));
  button.append(crop, body);
  card.append(button);
  if (face.error) card.append(el("p", face.error, "face-detail danger-text"));
  if (face.conflict_with != null) card.append(el("p", `Duplicate identity retained for face ${face.conflict_with}.`, "face-detail"));
  const candidates = Array.isArray(face.candidates) ? face.candidates : [];
  if (candidates.length || face.status === "possible" && face.identity) {
    const detail = el("details", null, "face-detail");
    detail.append(el("summary", "Match candidates"));
    if (face.status === "possible" && face.identity) detail.append(el("p", `Possible candidate: ${face.identity}`));
    for (const candidate of candidates) {
      detail.append(el("p", `${candidate.name} · cosine similarity ${score(candidate.score)}`));
    }
    card.append(detail);
  }
  return card;
}

export function mountAnalysis(state, actions) {
  const image = $("group-photo");
  const viewport = $("photo-viewport");
  const stage = $("photo-stage");
  const highlight = $("face-highlight");
  let lastResult, lastPhoto, scale = null, activeFace = null, annotated = false;

  function layout() {
    const { group, result } = state;
    if (!group?.width || !viewport.clientWidth || !viewport.clientHeight) return;
    const width = annotated ? result.image_width : group.width;
    const height = annotated ? result.image_height : group.height;
    const fit = Math.min(viewport.clientWidth / width, viewport.clientHeight / height, 1);
    const ratio = scale ?? fit;
    image.style.width = `${Math.round(width * ratio)}px`;
    image.style.height = `${Math.round(height * ratio)}px`;
    stage.style.setProperty("--photo-height", image.style.height);
    $("zoom-level").textContent = `${Math.round(ratio * 100)}%`;
  }

  function updateHighlight(scroll = false) {
    const result = state.result;
    const box = result?.results[activeFace]?.bbox;
    // The unannotated preview and analyzed raster use the same oriented frame.
    highlight.hidden = !validBox(box);
    if (highlight.hidden) return;
    const [x, y, width, height] = box;
    highlight.style.left = `${100 * x / result.image_width}%`;
    highlight.style.top = `${100 * y / result.image_height}%`;
    highlight.style.width = `${100 * width / result.image_width}%`;
    highlight.style.height = `${100 * height / result.image_height}%`;
    if (scroll) highlight.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  function renderViewer() {
    const { group, result } = state;
    const source = annotated && result ? result.annotated_image : group?.url;
    if (source && image.getAttribute("src") !== source) image.src = source;
    if (!source) image.removeAttribute("src");
    image.alt = annotated ? "Annotated group photograph" : "Original group photograph";
    $("empty-photo-picker").hidden = Boolean(group);
    viewport.hidden = $("viewer-tools").hidden = !source;
    $("choose-photo").textContent = group ? "Replace photo" : "Choose photo";
    $("view-original").setAttribute("aria-pressed", String(!annotated));
    $("view-annotated").setAttribute("aria-pressed", String(annotated));
    $("view-annotated").disabled = !result;
    $("photo-info").textContent = group
      ? `${group.file.name}${group.width ? ` · ${group.width} × ${group.height} px preview` : " · Reading image…"}`
      : "";
    layout();
    updateHighlight();
  }

  function selectFace(index) {
    activeFace = index;
    for (const button of $("result-list").querySelectorAll(".face-row")) {
      button.setAttribute("aria-pressed", String(Number(button.dataset.index) === index));
    }
    updateHighlight(true);
  }

  async function renderCrops(result, photo) {
    if (!result.results.length) return;
    const source = new Image();
    const useOriginal = photo.width === result.image_width && photo.height === result.image_height;
    source.src = useOriginal ? photo.url : result.annotated_image;
    try { await source.decode(); } catch { return; }
    if (state.result !== result) return;
    const canvases = $("result-list").querySelectorAll("canvas");
    result.results.forEach((face, index) => {
      if (!validBox(face.bbox)) return;
      const [x, y, w, h] = face.bbox;
      const factor = CROP_SIZE / Math.max(w, h);
      canvases[index].getContext("2d")?.drawImage(source, x, y, w, h,
        (CROP_SIZE - w * factor) / 2, (CROP_SIZE - h * factor) / 2, w * factor, h * factor);
    });
  }

  function render() {
    const { result, group, status, operation } = state;
    const processing = operation?.kind === "analysis";
    $("analyze-photo").disabled = !actions.canAnalyze();
    $("analyze-photo").textContent = processing ? "Analyzing…" : "Analyze photo";
    $("choose-photo").disabled = $("empty-photo-picker").disabled = Boolean(operation);
    $("results-content").hidden = !result;
    $("results-placeholder").hidden = Boolean(result);
    $("results-placeholder-title").textContent = processing ? "Analysis in progress" : "No analysis yet";
    $("results-placeholder-note").textContent = processing ? "Detecting faces and comparing them with enrolled references."
      : !state.connected ? "Connect to the local backend to run analysis."
      : !status?.engine_initialized ? "A recognition engine is required."
      : !status.gallery_loaded ? "Enroll reference identities in Gallery to enable analysis."
      : "Run analysis to see detected faces and match details.";
    if (group !== lastPhoto || result !== lastResult) {
      if (group?.url !== lastPhoto?.url) scale = null;
      if (result !== lastResult) {
        activeFace = null;
        annotated = Boolean(result);
      }
      lastPhoto = group;
      lastResult = result;
      $("result-list").replaceChildren();
      if (result) {
        tiles($("result-counts"), summaryEntries(result));
        $("result-time").textContent = ms(result.timings_ms?.backend_total);
        $("result-list").append(...result.results.map((face, i) => resultCard(face, i, selectFace)));
        if (!result.results.length) $("result-list").append(el("p", "No faces were detected.", "empty"));
        renderCrops(result, group);
      }
    }
    renderViewer();
  }

  new ResizeObserver(layout).observe(viewport);
  for (const id of ["choose-photo", "empty-photo-picker"]) $(id).addEventListener("click", () => $("group-file").click());
  $("group-file").addEventListener("change", event => {
    actions.choose(event.target.files[0]);
    event.target.value = "";
  });
  const drop = $("drop-zone");
  drop.addEventListener("dragover", event => {
    event.preventDefault();
    if (!state.operation) drop.classList.add("dragging");
  });
  drop.addEventListener("dragleave", event => {
    if (!drop.contains(event.relatedTarget)) drop.classList.remove("dragging");
  });
  drop.addEventListener("drop", event => {
    event.preventDefault();
    drop.classList.remove("dragging");
    if (event.dataTransfer.files.length) actions.choose(event.dataTransfer.files[0]);
  });
  $("analyze-photo").addEventListener("click", actions.run);
  for (const [id, value] of [["view-original", false], ["view-annotated", true]]) {
    $(id).addEventListener("click", () => { annotated = value; renderViewer(); });
  }
  for (const [id, value] of [["fit-photo", null], ["reset-zoom", 1]]) {
    $(id).addEventListener("click", () => { scale = value; layout(); viewport.scrollTo(0, 0); });
  }
  for (const [id, factor] of [["zoom-in", 1.25], ["zoom-out", 0.8]]) {
    $(id).addEventListener("click", () => {
      const current = Number.parseFloat(image.style.width) / (annotated ? state.result.image_width : state.group.width);
      scale = Math.min(4, Math.max(0.025, current * factor));
      layout();
    });
  }
  for (const format of ["jpeg", "csv"]) {
    $(`export-${format}`).addEventListener("click", () => {
      if (state.result && !state.operation) exportAnalysis(state.result, format);
    });
  }
  return render;
}
