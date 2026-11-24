import { post, request, validStatus } from "./api.js";
import { mountGallery } from "./gallery.js";
import { mountAnalysis, validateAnalysis } from "./analysis.js";
import { mountDashboard } from "./dashboard.js";
import { mountEvaluation } from "./evaluation.js";
import { mountSettings, referenceSelectionPolicy } from "./settings.js";
import { $, notice, busyNotice, setStatus } from "./ui.js";

const state = {
  workspace: "dashboard",
  connected: null,
  health: null,
  status: null,
  pipeline: null,
  operation: null,
  error: "",
  galleryFiles: [],
  galleryUpload: null,
  referenceSelectionPolicy: referenceSelectionPolicy(),
  group: null,
  result: null,
};
const mib = 1024 * 1024;
let previewRequest;

function render() {
  const { connected, status, health, operation } = state;
  const ready = status?.engine_initialized ?? health?.engine_initialized;
  const engine = operation?.kind === "pipeline" || ready == null && connected === null ? "Initializing"
    : !connected || ready == null ? "Engine status unavailable"
    : ready ? "Engine ready" : "Engine unavailable";
  const engineTone = connected && ready && operation?.kind !== "pipeline" ? "green" : connected && ready === false ? "red" : "";
  setStatus($("backend-status"), connected == null ? "Connecting" : connected ? "Backend connected" : "Backend unavailable",
    connected == null ? "" : connected ? "green" : "red");
  setStatus($("engine-status"), engine, engineTone);
  setStatus($("sidebar-engine"), engine, engineTone);
  $("sidebar-pipeline").textContent = status?.active_pipeline_label ?? health?.active_pipeline_id ?? "Pipeline unavailable";
  renderDashboard(engine);
  const policy = status?.image_upload;
  $("gallery-upload-policy").textContent = policy
    ? `${policy.formats.map(format => format === "HEIF" ? "HEIC/HEIF" : format).join(", ")} · Up to ${policy.max_bytes / mib} MiB per image`
    : "Image limits are available when the backend connects.";
  $("group-upload-policy").textContent = policy
    ? `or choose a file · Up to ${policy.max_bytes / mib} MiB` : "or choose a file";
  const warning = status?.fallback_reason ?? status?.startup_error ?? health?.fallback_reason ?? health?.startup_error;
  notice($("system-message"), state.error || warning, state.error ? "error" : "warning");
  for (const button of document.querySelectorAll("[data-refresh]")) button.disabled = Boolean(operation);
  renderGallery();
  renderAnalysis();
  renderEvaluation();
  renderSettings();
}

function adoptStatus(next) {
  validStatus(next);
  const previous = state.status;
  if (previous && (next.gallery_revision !== previous.gallery_revision
    || next.active_pipeline_id !== previous.active_pipeline_id || !next.engine_initialized || !next.gallery_loaded)) {
    state.result = null;
  }
  if (previous?.active_pipeline_id !== next.active_pipeline_id) {
    state.galleryUpload = null;
    notice($("gallery-message"));
    notice($("analysis-message"));
  } else if (state.operation?.kind === "refresh") {
    state.galleryUpload = null;
    notice($("gallery-message"));
  }
  state.status = next;
}

async function syncStatus() {
  const [health, status, pipeline] = await Promise.allSettled([
    request("/health").then(data => {
      if (typeof data.engine_initialized !== "boolean") throw new Error("Invalid engine health response.");
      return data;
    }),
    request("/status").then(validStatus),
    request("/pipeline").then(data => {
      if (!Array.isArray(data.available_pipelines)
          || !data.available_pipelines.every(item => typeof item.id === "string" && typeof item.label === "string")) {
        throw new Error("Invalid pipeline metadata.");
      }
      return data;
    }),
  ]);
  const replies = [health, status, pipeline];
  state.connected = replies.some(reply => reply.status === "fulfilled" || reply.reason?.status);
  state.health = health.status === "fulfilled" ? health.value : null;
  state.pipeline = pipeline.status === "fulfilled" ? pipeline.value : null;
  if (status.status === "fulfilled") adoptStatus(status.value);
  else state.status = null;
  if (state.result && (!state.status || state.result.pipeline_id !== state.status.active_pipeline_id
    || state.result.gallery_revision !== state.status.gallery_revision)) {
    state.result = null;
  }
  const failure = replies.find(reply => reply.status === "rejected");
  state.error = state.connected ? failure?.reason.message ?? "" : "The local FaceFind backend is unavailable. Start the app and refresh status.";
}

async function operate(kind, target, label, work) {
  if (state.operation) return;
  const started = performance.now();
  state.operation = { kind };
  const stopTimer = target ? busyNotice(target, label, started) : null;
  render();
  try {
    return await work(started);
  } catch (error) {
    if (error.disconnected || error.status === 409 || !error.status) await syncStatus();
    if (target) notice(target, error.message, "error");
    else state.error = error.message;
  } finally {
    stopTimer?.();
    state.operation = null;
    render();
  }
}

const refresh = () => operate("refresh", null, "", syncStatus);
function chooseGallery(files) {
  if (state.operation || !files.length) return;
  state.galleryFiles = files;
  notice($("gallery-message"));
  render();
}

function uploadGallery() {
  if (!state.connected || !state.status?.engine_initialized || !state.galleryFiles.length) return;
  const body = new FormData();
  for (const file of state.galleryFiles) body.append("files", file);
  body.append("reference_selection_policy", state.referenceSelectionPolicy);
  const target = $("gallery-message");
  operate("gallery", target, "Uploading and enrolling reference photos…", async () => {
    const data = await post("/gallery", body, state.status);
    adoptStatus(data);
    state.galleryUpload = data.upload_summary;
    state.galleryFiles = [];
    notice(target, data.message, data.build_succeeded ? "success" : "warning");
  });
}

function reviewReference(item, faceIndex) {
  const selecting = faceIndex != null;
  const target = $("gallery-message");
  operate("review", target, selecting ? "Enrolling selected face…" : "Skipping reference…", async () => {
    const data = await post(selecting ? "/gallery/select" : "/gallery/skip", {
      pending_id: item.pending_id, face_index: faceIndex, expected_identity: item.identity,
    }, state.status);
    adoptStatus(data);
    render();
    notice(target, data.message, "success");
  });
}

function reviewAutomaticReference(item, faceIndex) {
  return operate("automatic-review", $("automatic-message"), "Saving reference selection…", async () => {
    const data = await post(`/gallery/automatic/${item.pending_id}`, {
      revision: item.revision, face_index: faceIndex,
    }, state.status);
    adoptStatus(data);
    notice($("gallery-message"), data.message, "success");
    return true;
  });
}

function clearGallery() {
  if (state.operation || !window.confirm("Clear the reference gallery for the active pipeline?")) return;
  const target = $("gallery-message");
  operate("clear", target, "Clearing gallery…", async () => {
    const data = await post("/gallery/clear", null, state.status);
    adoptStatus(data);
    state.galleryUpload = null;
    state.galleryFiles = [];
    notice(target, data.message, "success");
  });
}

async function choosePhoto(file) {
  if (!file || state.operation) return;
  previewRequest?.abort();
  state.result = null;
  state.group = null;
  const target = $("analysis-message");
  notice(target);
  const limit = state.status?.image_upload?.max_bytes;
  if (!file.size || limit && file.size > limit) {
    notice(target, file.size ? `File exceeds ${limit / mib} MiB.` : "Choose a nonempty image.", "error");
    render();
    return;
  }
  const controller = new AbortController();
  previewRequest = controller;
  const group = { file, url: null, width: null, height: null };
  state.group = group;
  render();
  try {
    const body = new FormData();
    body.append("file", file);
    const data = await request("/image/preview", { method: "POST", body, signal: controller.signal });
    if (state.group !== group) return;
    if (typeof data.preview !== "string" || !data.preview.startsWith("data:image/png;base64,")
        || !Number.isInteger(data.width) || data.width < 1 || !Number.isInteger(data.height) || data.height < 1) {
      throw new Error("The backend returned an invalid image preview.");
    }
    Object.assign(group, { url: data.preview, width: data.width, height: data.height });
  } catch (error) {
    if (state.group !== group) return;
    state.group = null;
    notice(target, error.message, "error");
  }
  render();
}

function canAnalyze() {
  const status = state.status;
  return Boolean(state.connected && !state.operation && state.group?.width
    && status?.engine_initialized && status.gallery_loaded && status.gallery_pipeline_id === status.active_pipeline_id);
}

function runAnalysis() {
  if (!canAnalyze()) return;
  const target = $("analysis-message");
  operate("analysis", target, "Detecting and recognizing faces…", async started => {
    const body = new FormData();
    body.append("file", state.group.file);
    const data = validateAnalysis(await post("/analyze", body, state.status), state.status);
    state.result = { ...data, apiLatency: performance.now() - started };
    notice(target, `Analysis complete · ${data.face_count} detected`, "success");
  });
}

function switchPipeline(id) {
  if (!id || id === state.status?.active_pipeline_id) return;
  const target = $("pipeline-message");
  operate("pipeline", target, "Initializing local models…", async () => {
    const data = await post("/pipeline", { pipeline_id: id }, state.status);
    adoptStatus(data);
    await syncStatus();
    notice(target, data.gallery_loaded ? "Pipeline ready. Cached gallery restored." : "Pipeline ready. Enroll references for this pipeline.", "success");
  });
}

function navigate(focus = false) {
  const target = location.hash.slice(1);
  if (target === "main") return;
  const name = { dashboard: "Dashboard", gallery: "Gallery", analyze: "Analyze", evaluation: "Evaluation" };
  state.workspace = Object.hasOwn(name, target) ? target : "dashboard";
  for (const page of Object.keys(name)) $(`workspace-${page}`).hidden = page !== state.workspace;
  for (const link of document.querySelectorAll("[data-workspace]")) {
    if (link.dataset.workspace === state.workspace) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  $("workspace-name").textContent = name[state.workspace];
  if (focus) {
    $("main").focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }
}

const renderGallery = mountGallery(state, {
  choose: chooseGallery, upload: uploadGallery, review: reviewReference,
  reviewAutomatic: reviewAutomaticReference, clear: clearGallery, refresh,
});
const renderAnalysis = mountAnalysis(state, { choose: choosePhoto, canAnalyze, run: runAnalysis });
const renderSettings = mountSettings(state, switchPipeline);
const renderDashboard = mountDashboard(state);
const renderEvaluation = mountEvaluation(state);
for (const button of document.querySelectorAll("[data-refresh]")) button.addEventListener("click", refresh);
window.addEventListener("hashchange", () => navigate(true));
navigate();
refresh();
