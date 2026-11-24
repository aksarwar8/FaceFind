import { el, ms, score, tiles, details } from "./ui.js";
import { OUTCOMES } from "./evaluation-core.js";

export const FILTERS = {
  all: "All", correct_known: "Correctly identified", wrong_known: "Wrong identity",
  correct_unknown: "Correct Unknown", false_accept: "False acceptance", rejected_known: "Incorrect rejection",
  missed: "Missed face", false_detection: "False detection", processing_failure: "Processing failure",
  unverified: "Not verified",
};
export const filteredRows = (rows, filter) => rows.filter(row =>
  filter === "all" || row.outcome === filter || filter === "missed" && row.outcome.startsWith("missed_"));
export const formatMetric = metric => metric?.value == null ?
  (metric ? `N/A (${metric.numerator}/${metric.denominator})` : "N/A")
  : `${(metric.value * 100).toFixed(1)}% (${metric.numerator}/${metric.denominator})`;

function image(url, alt) {
  const photo = el("img");
  photo.src = url;
  photo.alt = alt;
  return photo;
}

function boxPercent(box, snapshot) {
  return box.map((value, index) => 100 * value / (index % 2 ? snapshot.image_height : snapshot.image_width));
}

function crop(url, snapshot, box) {
  const node = el("span", null, "evaluation-crop");
  const [x, y, w, h] = boxPercent(box, snapshot);
  if (!(w > 0 && h > 0)) return node;
  const photo = image(url, "");
  photo.style.width = `${10000 / w}%`;
  photo.style.height = `${10000 / h}%`;
  photo.style.left = `${-100 * x / w}%`;
  photo.style.top = `${-100 * y / h}%`;
  node.style.aspectRatio = `${box[2]} / ${box[3]}`;
  node.append(photo);
  return node;
}

export const rowLabel = row => row.prediction ? `Face ${row.prediction.face_number}` : `Missed ${row.actual.id}`;
export const rowBox = row => row.prediction?.bbox ?? row.actual?.bbox;
const correct = row => row.outcome === "correct_known" || row.outcome === "correct_unknown";
const tone = row => row.outcome === "unverified" ? "neutral" : correct(row) ? "good" : "bad";

export function viewer(url, snapshot, rows, selected, select) {
  const stage = el("div", null, "evaluation-stage");
  if (!url) { stage.textContent = "Original photo preview unavailable."; return stage; }
  stage.append(image(url, "Original group photo with evaluation face boxes"));
  for (const row of rows) {
    const box = rowBox(row);
    if (!box) continue;
    const button = el("button", null, `evaluation-box ${tone(row)}`);
    button.append(el("span", row.prediction?.face_number ?? "M"));
    button.type = "button";
    button.setAttribute("aria-label", `${rowLabel(row)}: ${OUTCOMES[row.outcome]}`);
    button.setAttribute("aria-pressed", String(selected === row.key));
    const [x, y, w, h] = boxPercent(box, snapshot);
    Object.assign(button.style, { left: `${x}%`, top: `${y}%`, width: `${w}%`, height: `${h}%` });
    button.addEventListener("click", () => select(row.key, "box"));
    stage.append(button);
  }
  return stage;
}

export function faceCard(row, snapshot, url, selected, select) {
  const card = el("button", null, `evaluation-face ${tone(row)}`);
  card.type = "button";
  card.setAttribute("aria-pressed", String(row.key === selected));
  card.addEventListener("click", () => select(row.key, "row"));
  if (url && rowBox(row)) card.append(crop(url, snapshot, rowBox(row)));
  const copy = el("span", null, "evaluation-face-copy");
  const p = row.prediction;
  const prediction = p?.status === "recognised" ? p.identity : p?.status === "possible" ? `Possible: ${p.identity}`
    : p?.status === "error" ? "Processing error" : p ? "Unknown" : "No detection";
  const actual = row.actual ? `${row.actual.identity} (${row.actual.known ? "enrolled" : "not enrolled"})`
    : row.outcome === "false_detection" ? "Not a real face" : "Not verified";
  copy.append(el("strong", `${rowLabel(row)} - ${OUTCOMES[row.outcome]}`),
    el("span", `Predicted: ${prediction}  |  Actual: ${actual}`));
  if (p) copy.append(el("small", `Detection score ${score(p.confidence)} | Cosine similarity ${score(p.best_score)}`));
  if (row.iou != null) copy.append(el("small", `Matched IoU ${row.iou.toFixed(3)}`));
  card.append(copy);
  return card;
}

function panel(title, entries) {
  const root = el("article", null, "card padded");
  const values = el("div", null, "result-counts");
  tiles(values, entries);
  root.append(el("h2", title), values);
  return root;
}

export function performance(snapshot) {
  const root = el("article", null, "card padded");
  const list = el("dl", null, "details-grid");
  const timings = snapshot.timings_ms;
  details(list, [
    ["Pipeline", snapshot.pipeline_id], ["Gallery size", snapshot.gallery_identities.length],
    ["Working image", `${snapshot.image_width} x ${snapshot.image_height}`],
    ["Backend total", ms(timings?.backend_total)], ["API round trip", ms(snapshot.apiLatency)],
    ["Detection", ms(timings?.detection)], ["Alignment", ms(timings?.face_alignment_total)],
    ["Embedding", ms(timings?.face_embedding_total)], ["Matching", ms(timings?.identity_matching_total)],
    ["Annotation and encoding", ms(timings?.annotation_and_rendering)],
    ["Recognition threshold / margin", `${snapshot.recognition_rules?.recognition_threshold ?? "N/A"} / ${snapshot.recognition_rules?.margin_threshold ?? "N/A"}`],
  ]);
  root.append(el("h2", "Inference provenance"), list);
  return root;
}

export function evaluationPanels(outcome, save) {
  const section = el("div", null, "evaluation-panels");
  const { counts: c, detection: d, metrics: m, known, unknown } = outcome;
  section.append(panel("Evaluation summary", [
    ["Ground-truth faces", known + unknown], ["Detected faces", outcome.detected],
    ["Correct identifications", c.correct_known], ["Wrong identities", c.wrong_known],
    ["Correct Unknown", c.correct_unknown], ["False acceptances", c.false_accept],
    ["Missed faces", c.missed_known + c.missed_unknown], ["False detections", c.false_detection],
    ["Overall correct outcomes", formatMetric(m.overall)],
  ]));
  section.append(d ? panel(d.method === "iou" ? "Detection performance (IoU >= 0.50)" : "Detection performance (manual correspondence)", [
    ["TP", d.tp], ["FP", d.fp], ["FN", d.fn],
    ["Precision", formatMetric(d.precision)], ["Recall", formatMetric(d.recall)], ["F1", formatMetric(d.f1)],
  ]) : panel("Detection performance", [["Scope", "Complete the manual review to report detection metrics"]]));
  section.append(panel(outcome.complete ? "End-to-end recognition" : "Recognition on verified subset", [
    ["Known identification", formatMetric(m.known)], ["Unknown rejection", formatMetric(m.unknown)],
    ["False acceptance", formatMetric(m.falseAcceptance)], ["Known identification failure", formatMetric(m.knownFailure)],
    ["Conditional known identification", formatMetric(m.conditionalKnown)],
    ["Conditional Unknown rejection", formatMetric(m.conditionalUnknown)], ["Processing failures", c.processing_failure],
  ]));
  const matrix = el("article", null, "card padded evaluation-matrix");
  const table = el("table");
  const head = el("thead"), tr = el("tr");
  for (const title of ["Ground truth", "Correct identity", "Incorrect accepted identity", "Rejected as Unknown", "Missed", "Processing failure"]) {
    const th = el("th", title);
    th.scope = "col";
    tr.append(th);
  }
  head.append(tr);
  const body = el("tbody");
  for (const [title, values] of [["Actual Known", outcome.matrix.known], ["Actual Unknown", outcome.matrix.unknown]]) {
    const row = el("tr"), th = el("th", title);
    th.scope = "row";
    row.append(th, ...values.map(value => el("td", value ?? "N/A")));
    body.append(row);
  }
  table.append(head, body);
  const exports = el("div", null, "toolbar wrap");
  for (const [type, title] of [["json", "Download evaluation JSON"], ["csv", "Download per-face CSV"], ["jpg", "Download evaluation JPEG"]]) {
    const button = el("button", title);
    button.type = "button";
    button.addEventListener("click", () => save(type));
    exports.append(button);
  }
  matrix.append(el("h2", "Open-set outcome matrix"), table,
    el("p", "Unmatched detections appear separately. Possible labels are tentative and scored as unassigned."), exports);
  section.append(matrix);
  return section;
}
