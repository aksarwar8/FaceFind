import { summaryEntries } from "./analysis.js";
import { enrolledRows, referencePortrait } from "./gallery.js";
import { $, el, icon, ms, tiles } from "./ui.js";

function emptyPreview(title, note, symbol) {
  const panel = el("div", null, "empty-panel");
  const mark = el("span", null, "tile-icon");
  mark.append(icon(symbol));
  const copy = el("div");
  copy.append(el("strong", title), el("p", note));
  panel.append(mark, copy);
  return panel;
}

export function mountDashboard(state) {
  let lastStatus;
  let lastResult;

  return engine => {
    const { status, result } = state;
    $("metric-people").textContent = status?.people_count ?? "—";
    $("metric-pending").textContent = status?.pending.length ?? "—";
    $("metric-engine").textContent = engine;
    $("metric-pipeline").textContent = status?.active_pipeline_label ?? "Pipeline unavailable";
    $("metric-gallery-note").textContent = !status ? "Status unavailable" : status.gallery_loaded ? "Gallery ready" : "Gallery empty";
    $("metric-time").textContent = result ? ms(result.timings_ms?.backend_total) : "—";

    if (status !== lastStatus) {
      const target = $("dashboard-gallery");
      const enrolled = enrolledRows(status);
      const list = el("ul", null, "gallery-preview");
      for (const item of enrolled.slice(0, 5)) {
        const entry = el("li");
        entry.append(referencePortrait(item), el("span", item.name));
        list.append(entry);
      }
      target.replaceChildren(enrolled.length ? list : emptyPreview(
        status ? "No enrolled references" : "Gallery status unavailable",
        status ? "Build a gallery to see reference portraits here." : "Check the local backend status.",
        "people",
      ));
      if (enrolled.length > 5) target.append(el("p", `${enrolled.length - 5} more in Gallery Manager`, "preview-note"));
      lastStatus = status;
    }

    if (result !== lastResult) {
      const target = $("dashboard-result");
      if (result) {
        const preview = el("a", null, "analysis-preview");
        preview.href = "#analyze";
        preview.setAttribute("aria-label", "Open the latest analysis");
        const image = el("img");
        image.alt = "Annotated group photo preview";
        image.src = result.annotated_image;
        const summary = el("div", null, "analysis-preview-summary");
        const counts = el("div", null, "result-counts");
        tiles(counts, summaryEntries(result));
        summary.append(el("strong", result.filename), counts);
        preview.append(image, summary);
        target.replaceChildren(preview);
      } else {
        target.replaceChildren(emptyPreview("No current analysis", "Analyze a group photo to see its preview and summary.", "photo"));
      }
      lastResult = result;
    }
  };
}
