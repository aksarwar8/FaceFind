import { $, el, notice, tiles } from "./ui.js";
import { downloadBlob } from "./exports.js";
import { analysisSnapshot, currentEvaluation, evaluate, galleryMap, identityKey, MAX_FACES, OUTCOMES } from "./evaluation-core.js";
import { MAX_TRUTH_BYTES, TEMPLATE, mapBox, parseGroundTruth } from "./evaluation-import.js";
import { evaluationCsv, evaluationReport } from "./evaluation-export.js";
import { FILTERS, filteredRows, formatMetric, faceCard, viewer, performance, evaluationPanels, rowBox, rowLabel } from "./evaluation-view.js";

const manual = (source = "Manual review") => ({ mode: "manual", source, labels: new Map(), missed: [], complete: false });
const option = (value, caption) => Object.assign(el("option", caption), { value });
const predictionsOnly = snapshot => snapshot.results.map(prediction => ({
  key: `p-${prediction.face_number}`, prediction, actual: null, outcome: "unverified",
}));
const input = (type, name) => {
  const control = el("input");
  control.type = type;
  control.setAttribute("aria-label", name);
  return control;
};

export function mountEvaluation(state) {
  let snapshot, original, data, selected, missedSequence = 0;
  let outcome, importGeneration = 0;
  const filter = $("evaluation-filter");
  filter.append(...Object.entries(FILTERS).map(([value, title]) => option(value, title)));
  const message = $("evaluation-message");
  const fileInput = $("evaluation-file");
  const fileButton = $("evaluation-import");
  const template = $("evaluation-template");
  const start = $("evaluation-manual");
  const target = $("evaluation-content");

  function update(change) {
    change();
    data.complete = false;
    notice(message);
    redraw();
  }

  function choose(key, from) {
    selected = key;
    redraw();
    const panel = from === "box" ? $("evaluation-results") : $("evaluation-viewer");
    const current = panel.querySelector('[aria-pressed="true"]');
    current?.focus({ preventScroll: true });
    if (from === "box") current?.scrollIntoView({ block: "nearest" });
  }

  function setLabel(number, actual) {
    update(() => {
      if (actual) data.labels.set(number, actual);
      else data.labels.delete(number);
    });
  }

  function identityForm(heading, existing, missed, save) {
    const form = el("form", null, "evaluation-controls");
    form.append(el("h3", heading));
    const roster = data.roster;
    const kind = el("select");
    kind.setAttribute("aria-label", missed ? "Missed face identity type" : "Actual identity type");
    if (!missed) kind.append(option("", "Not verified"), option("false", "Not a real face"));
    if (roster) kind.append(option("roster", "Select an imported face"));
    else kind.append(option("known", "Enrolled identity"), option("unknown", "Person not in the reference gallery"));
    kind.value = existing?.falseDetection ? "false" : existing ? (roster ? "roster" : existing.known ? "known" : "unknown")
      : missed ? (roster ? "roster" : "known") : "";
    const select = el("select");
    select.setAttribute("aria-label", "Imported face ID");
    select.append(option("", "Choose imported face"));
    const used = new Set([...data.labels.values(), ...data.missed].map(item => item.id).filter(id => id !== existing?.id));
    select.append(...(roster ?? []).filter(face => !used.has(face.id)).map(face =>
      option(face.id, `${face.id} - ${face.identity}`)));
    select.value = existing?.id ?? "";
    const identities = galleryMap(snapshot);
    const name = input("text", "Actual identity name");
    name.maxLength = 240;
    name.value = existing?.identity ?? "";
    const datalist = el("datalist");
    datalist.id = missed ? "evaluation-missed-gallery" : "evaluation-face-gallery";
    datalist.append(...[...identities.values()].map(value => option(value, value)));
    const box = ["X", "Y", "Width", "Height"].map(label => {
      const field = input("number", `Missed box ${label}`);
      field.placeholder = label;
      field.step = "any";
      field.required = true;
      return field;
    });
    function setFields() {
      name.hidden = !["known", "unknown"].includes(kind.value);
      name.placeholder = kind.value === "known" ? "Search enrolled names" : "Optional actual name";
      if (kind.value === "known") name.setAttribute("list", datalist.id);
      else name.removeAttribute("list");
      select.hidden = kind.value !== "roster";
    }
    kind.addEventListener("change", setFields);
    setFields();
    form.append(kind, select, name, datalist);
    if (missed) {
      const fields = el("div", null, "evaluation-box-fields");
      fields.append(...box);
      form.append(el("p", "Enter x, y, width and height in the working image pixels shown below."), fields);
    }
    const button = el("button", missed ? "Add missed face" : "Save ground truth");
    button.type = "submit";
    form.append(button);
    form.addEventListener("submit", event => {
      event.preventDefault();
      try {
        let actual;
        if (kind.value === "roster") {
          actual = roster.find(face => face.id === select.value && !used.has(face.id));
          if (!actual) throw new Error("Choose an unused imported face ID.");
        } else if (kind.value === "known") {
          const enrolled = identities.get(identityKey(name.value));
          if (!enrolled) throw new Error("Select a name from the frozen gallery.");
          actual = { identity: enrolled, known: true };
        } else if (kind.value === "unknown") {
          if (identities.has(identityKey(name.value))) throw new Error("That name is enrolled in the frozen gallery.");
          actual = { identity: name.value.trim() || "Unknown person", known: false };
        } else if (kind.value === "false") actual = { falseDetection: true };
        if (missed) {
          if (data.missed.length + data.labels.size >= MAX_FACES) throw new Error("Manual review supports up to 256 actual faces.");
          const width = snapshot.image_width, height = snapshot.image_height;
          const bbox = mapBox(box.map(field => field.value === "" ? NaN : Number(field.value)), width, height, 1, width, height);
          const face = { ...actual, id: actual.id ?? `manual-${++missedSequence}`, bbox };
          update(() => { data.missed.push(face); selected = `g-${face.id}`; filter.value = "all"; });
        } else {
          save(actual);
        }
      } catch (error) { notice(message, error.message, "error"); }
    });
    return form;
  }

  function manualControls(row) {
    const controls = el("div", null, "evaluation-manual");
    if (!data || data.mode !== "manual") return controls;
    const complete = input("checkbox", "Complete manual ground truth");
    const used = new Set([...data.labels.values(), ...data.missed].map(face => face.id));
    complete.checked = data.complete;
    complete.disabled = outcome.verified !== outcome.detected || Boolean(data.roster?.some(face => !used.has(face.id)));
    complete.addEventListener("change", () => { data.complete = complete.checked; redraw(); });
    const label = el("label", null, "evaluation-attestation");
    label.append(complete, el("span", "I checked the entire photo and accounted for every real and missed face."));
    controls.append(label);
    if (data.roster) controls.append(el("p", `${data.roster.filter(face => !used.has(face.id)).length} imported faces still need manual correspondence.`));
    if (row?.prediction) {
      const num = row.prediction.face_number;
      controls.append(identityForm(`Actual identity for face ${num}`, data.labels.get(num), false, actual => {
        if (actual && !actual.falseDetection && !data.roster) actual.id = `face-${num}`;
        setLabel(num, actual);
      }));
    } else if (row?.actual) {
      const remove = el("button", "Remove missed annotation");
      remove.type = "button";
      remove.addEventListener("click", () => update(() => {
        data.missed = data.missed.filter(face => face.id !== row.actual.id);
        selected = null;
      }));
      controls.append(remove);
    }
    const add = el("details");
    add.append(el("summary", "Mark a missed face"), identityForm("Missed face", null, true));
    controls.append(add);
    return controls;
  }

  function redraw() {
    if (!snapshot) return;
    outcome = data ? evaluate(snapshot, data) : null;
    const rows = outcome?.rows ?? predictionsOnly(snapshot);
    const visible = filteredRows(rows, filter.value || "all");
    if (!visible.some(row => row.key === selected)) selected = visible[0]?.key;
    $("evaluation-viewer").replaceChildren(viewer(original, snapshot, visible, selected, choose));
    $("evaluation-results").replaceChildren(...(visible.length
      ? visible.map(row => faceCard(row, snapshot, original, selected, choose)) : [el("p", "No faces match this filter.")]));
    $("evaluation-manual-controls").replaceChildren(manualControls(visible.find(row => row.key === selected)));
    const status = !outcome ? "Not Evaluated" : outcome.complete ? "Evaluated" : "Verified subset";
    $("evaluation-status").textContent = status;
    $("evaluation-progress").textContent = outcome ? `${outcome.verified} of ${outcome.detected} detections verified` : "No ground truth supplied";
    $("evaluation-context").textContent = `${snapshot.filename} | ${snapshot.pipeline_id} | ${data?.source ?? "No ground truth"}`;
    $("evaluation-report").replaceChildren(...(outcome ? [evaluationPanels(outcome, saveReport)] : []), performance(snapshot));
    const latest = $("dashboard-evaluation");
    latest.replaceChildren(el("p", `${status} | ${snapshot.filename} | ${snapshot.pipeline_id}`));
    if (outcome) {
      const counts = el("div", null, "result-counts");
      tiles(counts, [["Overall correct", formatMetric(outcome.metrics.overall)],
        ["Known identification", formatMetric(outcome.metrics.known)], ["Unknown rejection", formatMetric(outcome.metrics.unknown)]]);
      latest.append(counts);
    }
  }

  function presentImport(parsed, file, truthHash) {
    const source = file.name.split(/[\\/]/).at(-1);
    data = parsed.manualRequired ? { ...manual(source), roster: parsed.faces, ...parsed }
      : { mode: "import", source, ...parsed };
    data.sourceHash = truthHash;
    const messages = [
      parsed.manualRequired && "Some boxes are missing or unreliable. Pair imported face IDs manually and mark any missed faces.",
      parsed.approximate && "These mapped composite boxes are approximate.",
      !parsed.approximate && !parsed.manualRequired && "Imported boxes are used as supplied.",
      ...parsed.warnings,
    ].filter(Boolean);
    notice(message, messages.join(" "), messages.length ? "warning" : "");
    filter.value = "all";
    selected = null;
    redraw();
  }

  async function saveReport(type) {
    if (!snapshot || !outcome || !currentEvaluation(snapshot, state.result, state.status)) return;
    const report = evaluationReport(snapshot, data, outcome);
    const name = snapshot.filename.replace(/\.[^.]+$/, "").replace(/[^\p{L}\p{N}_.-]/gu, "_");
    if (type === "json" || type === "csv") {
      const body = type === "json" ? JSON.stringify(report, null, 2) : "\uFEFF" + evaluationCsv(report);
      downloadBlob(new Blob([body], { type: type === "json" ? "application/json" : "text/csv" }), `${name}-evaluation.${type}`);
      return;
    }
    const frozen = snapshot;
    const photo = new Image();
    photo.src = original ?? "";
    try {
      await photo.decode();
      const canvas = document.createElement("canvas");
      canvas.width = photo.naturalWidth;
      canvas.height = photo.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(photo, 0, 0);
      const sx = canvas.width / frozen.image_width, sy = canvas.height / frozen.image_height;
      ctx.lineWidth = Math.max(2, canvas.width / 400);
      ctx.font = `bold ${Math.max(13, canvas.width / 100)}px sans-serif`;
      for (const row of report.rows) {
        const box = rowBox(row);
        if (!box) continue;
        const [x, y, w, h] = box.map((v, i) => v * (i % 2 ? sy : sx));
        const label = `${rowLabel(row)}: ${OUTCOMES[row.outcome]}`;
        ctx.strokeStyle = ["correct_known", "correct_unknown"].includes(row.outcome) ? "#39e3a0" : "#ffc466";
        ctx.strokeRect(x, y, w, h);
        const width = Math.min(ctx.measureText(label).width + 10, canvas.width);
        const tx = Math.max(0, Math.min(x, canvas.width - width)), ty = Math.max(18, y);
        ctx.fillStyle = "#111827";
        ctx.fillRect(tx, ty - 18, width, 23);
        ctx.fillStyle = "white";
        ctx.fillText(label, tx + 4, ty, width - 8);
      }
      canvas.toBlob(blob => {
        if (blob && frozen === snapshot && currentEvaluation(frozen, state.result, state.status)) downloadBlob(blob, `${name}-evaluation.jpg`);
      }, "image/jpeg", .92);
    } catch { notice(message, "The original preview could not be exported.", "error"); }
  }

  filter.addEventListener("change", redraw);
  start.addEventListener("click", () => {
    if (!snapshot || data?.mode === "manual") return;
    importGeneration++;
    data = manual();
    notice(message);
    redraw();
  });
  fileButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0], frozen = snapshot;
    fileInput.value = "";
    if (!file || !frozen) return;
    const generation = ++importGeneration;
    const isCurrent = () => generation === importGeneration && frozen === snapshot
      && currentEvaluation(frozen, state.result, state.status);
    try {
      if (file.size > MAX_TRUTH_BYTES) throw new Error("Ground truth exceeds 2 MiB.");
      const bytes = await file.arrayBuffer();
      const parsed = parseGroundTruth(new TextDecoder("utf-8", { fatal: true }).decode(bytes),
        file.name.split(".").at(-1).toLowerCase(), frozen);
      const digest = await crypto.subtle.digest("SHA-256", bytes);
      const truthHash = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");
      if (isCurrent()) presentImport(parsed, file, truthHash);
    } catch (error) {
      if (isCurrent()) notice(message, error.message, "error");
    }
  });
  template.addEventListener("click", () => downloadBlob(new Blob([TEMPLATE], { type: "text/csv" }), "facefind-ground-truth-template.csv"));

  return () => {
    const { result, status } = state;
    if (currentEvaluation(snapshot, result, status)) return;
    snapshot = original = data = selected = outcome = null;
    filter.value = "all";
    notice(message);
    const available = currentEvaluation(result, result, status) && Array.isArray(result.gallery_identities);
    target.hidden = !available;
    $("evaluation-empty").hidden = available;
    if (!available) {
      $("evaluation-status").textContent = "Not Evaluated";
      $("dashboard-evaluation").textContent = "Not Evaluated";
      for (const id of ["evaluation-viewer", "evaluation-results", "evaluation-manual-controls", "evaluation-report"]) $(id).replaceChildren();
      return;
    }
    snapshot = analysisSnapshot(result);
    original = state.group?.url;
    fileButton.disabled = result.results.length > MAX_FACES;
    if (fileButton.disabled) notice(message, "IoU import supports up to 256 predictions. Manual verification is available.", "warning");
    redraw();
  };
}
