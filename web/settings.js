import { $, details, el, ms } from "./ui.js";

const enrollmentPolicyKey = "facefind.referenceSelectionPolicy";

export function referenceSelectionPolicy() {
  try {
    return localStorage.getItem(enrollmentPolicyKey) === "always_review" ? "always_review" : "smart";
  } catch {
    return "smart";
  }
}

export function mountSettings(state, switchPipeline) {
  const dialog = $("settings");
  const select = $("pipeline-select");
  const enrollmentPolicy = $("reference-selection-policy");
  enrollmentPolicy.value = state.referenceSelectionPolicy;
  enrollmentPolicy.addEventListener("change", () => {
    state.referenceSelectionPolicy = enrollmentPolicy.value;
    try {
      localStorage.setItem(enrollmentPolicyKey, enrollmentPolicy.value);
    } catch {
      // The current page still uses the selected policy.
    }
  });
  let lastPipeline;

  for (const button of document.querySelectorAll("[data-settings]")) {
    button.addEventListener("click", () => dialog.showModal());
  }
  function toggleSwitch() {
    $("switch-pipeline").disabled = select.disabled || !select.value
      || select.value === state.status?.active_pipeline_id;
  }
  select.addEventListener("change", toggleSwitch);
  $("switch-pipeline").addEventListener("click", () => switchPipeline(select.value));

  return () => {
    const { pipeline, status, result, operation } = state;
    if (pipeline !== lastPipeline || !select.options.length) {
      const options = pipeline?.available_pipelines ?? [];
      select.replaceChildren(...options.map(item => {
        const option = el("option", item.label);
        option.value = item.id;
        return option;
      }));
      select.value = status?.active_pipeline_id ?? pipeline?.requested_pipeline_id ?? "";
      if (select.selectedIndex === -1 && options.length) select.selectedIndex = 0;
      lastPipeline = pipeline;
    }
    select.disabled = !state.connected || !status || Boolean(operation) || !select.options.length;
    toggleSwitch();
    const options = pipeline?.available_pipelines ?? [];
    const preferred = options.find(item => item.id === "scrfd_adaface");
    const requested = options.find(item => item.id === pipeline?.requested_pipeline_id);
    const initialized = options.filter(item => pipeline?.initialized_pipelines?.includes(item.id));
    details($("model-info"), [
      ["Preferred default", preferred?.label ?? "SCRFD-10G-KPS + AdaFace IR-18"],
      ["Active pipeline", status?.active_pipeline_label],
      ["Last requested pipeline", requested?.label],
      ["Detector", pipeline?.detector],
      ["Recognizer", pipeline?.recognizer],
      ["Embedding dimensions", pipeline?.embedding_dimensions],
      ["Processing device", pipeline?.device],
      ["Initialization", !status ? "Unavailable" : status.engine_initialized ? "Ready" : "Engine unavailable"],
      ["Initialized model bundles", initialized.map(item => item.label).join(", ") || "Unavailable"],
      ["Model initialization time", ms(pipeline?.model_initialization_ms)],
      ["Fallback reason", status?.fallback_reason || status?.startup_error || (status ? "None reported" : "Unavailable")],
    ]);
    const timing = result?.timings_ms;
    details($("performance"), [
      ["Detection", ms(timing?.detection)],
      ["Face alignment", ms(timing?.face_alignment_total)],
      ["Embedding extraction", ms(timing?.face_embedding_total)],
      ["Identity matching", ms(timing?.identity_matching_total)],
      ["Annotation and rendering", ms(timing?.annotation_and_rendering)],
      ["Backend processing", ms(timing?.backend_total)],
      ["API request latency", ms(result?.apiLatency)],
      ["Gallery embeddings reused", typeof timing?.gallery_embeddings_reused === "boolean"
        ? (timing.gallery_embeddings_reused ? "Yes" : "No") : "Unavailable"],
    ]);
  };
}
