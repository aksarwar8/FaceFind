import { createReferenceReview } from "./reference-review.js";
import { $, el, icon, notice } from "./ui.js";

export function enrolledRows(status) {
  const thumbnails = new Set(status?.thumbnail_indices);
  return (status?.identities ?? []).map((name, index) => ({
    name, status: "accepted", label: "Enrolled",
    thumbnail: thumbnails.has(index) && /^[a-f0-9]{32}$/.test(status.gallery_revision)
      ? `/api/gallery/thumbnail/${status.gallery_revision}/${index}` : null,
  }));
}

export function identityRows({ status }) {
  const pending = (status?.pending ?? []).map(item => ({
    name: item.identity, status: "pending", label: `${item.faces} detected faces`, pending: item,
  }));
  const rejected = (status?.gallery_build_errors ?? []).map(error => {
    const separator = error.indexOf(": ");
    return {
      name: separator < 0 ? "Reference image" : error.slice(0, separator) || "Unnamed file",
      status: "rejected",
      label: separator < 0 ? error : error.slice(separator + 2),
    };
  });
  return [...enrolledRows(status), ...pending, ...rejected];
}

export function referencePortrait(item) {
  const portrait = el("span", null, "reference-portrait");
  portrait.append(icon("people"));
  if (item.thumbnail) {
    const image = el("img");
    image.alt = "";
    image.loading = "lazy";
    image.addEventListener("error", () => image.remove(), { once: true });
    image.src = item.thumbnail;
    portrait.append(image);
  }
  return portrait;
}

function referenceCard(item, automatic, openReview, openPendingReview, activePendingId) {
  const card = el(item.pending ? "button" : "article", null, `identity identity-${item.status}`);
  const body = el(item.pending ? "span" : "div", null, "identity-label");
  body.append(el("strong", item.name), el("small", item.label));
  if (item.status === "accepted") card.append(referencePortrait(item));
  card.append(body);
  if (item.pending) {
    card.type = "button";
    card.dataset.pendingId = item.pending.pending_id;
    card.setAttribute("aria-label", `Review ${item.name}, ${item.label}`);
    card.setAttribute("aria-controls", "pending-section");
    card.setAttribute("aria-pressed", String(item.pending.pending_id === activePendingId));
    card.addEventListener("click", () => openPendingReview(item.pending.pending_id));
    return card;
  }
  if (automatic) {
    const label = automatic.selected_face_index === automatic.automatic_selection.selected_face_index
      ? "Review automatic selection" : "Review face selection";
    const button = el("button", label);
    button.setAttribute("aria-label", `${label} for ${item.name}`);
    button.addEventListener("click", () => openReview(automatic));
    body.append(button);
  }
  return card;
}

export function mountGallery(state, actions) {
  let rows = [];
  let previousStatus, previousUpload, review;
  let activePendingId, activePipeline;
  let previousPending = [];
  let automaticReview, automaticRevision;

  function renderRows() {
    const query = $("identity-search").value.trim().toLocaleLowerCase();
    const filter = $("identity-filter").value;
    const visible = rows.filter(item => (filter === "all" || item.status === filter)
      && String(item.name).toLocaleLowerCase().includes(query));
    const container = $("identity-grid");
    const automatic = new Map((state.status?.automatic_references ?? []).map(item => [item.identity, item]));
    container.replaceChildren();
    for (const [key, label] of [["accepted", "Enrolled"], ["pending", "Pending manual review"], ["rejected", "Rejected references"]]) {
      const group = visible.filter(item => item.status === key);
      if (!group.length) continue;
      const section = el("section", null, `reference-group group-${key}`);
      const title = el("h3", label);
      title.append(el("span", group.length, "tag"));
      const grid = el("div", null, key === "accepted" ? "identity-grid" : "reference-list");
      grid.append(...group.map(item => referenceCard(item, automatic.get(item.name), openAutomaticReview,
        openPendingReview, activePendingId)));
      section.append(title, grid);
      container.append(section);
    }
    if (!visible.length) {
      const panel = el("div", null, "empty-panel");
      const mark = el("span", null, "tile-icon");
      mark.append(icon("gallery-icon"));
      const copy = el("div");
      const loading = state.operation?.kind === "gallery";
      const title = loading ? "Enrolling references" : !state.status ? "Gallery status unavailable"
        : rows.length ? "No matching references" : "Your reference gallery is empty";
      const note = loading ? "FaceFind is checking the selected photos."
        : !state.status ? "Refresh status to check the local backend."
        : rows.length ? "Try a different name or enrollment status."
        : "Select a named reference folder or choose individual photos to start enrollment.";
      copy.append(el("strong", title), el("p", note));
      panel.append(mark, copy);
      container.append(panel);
    }
  }

  function updateReviewControls() {
    review?.setBusy(Boolean(state.operation || !state.connected));
    automaticReview?.setBusy(Boolean(state.operation || !state.connected));
  }

  function closeAutomaticReview() {
    automaticReview?.dispose();
    automaticReview = null;
    automaticRevision = null;
    $("automatic-item").replaceChildren();
    notice($("automatic-message"));
    $("automatic-section").hidden = true;
  }

  function openAutomaticReview(item) {
    if (state.operation || !state.connected) return;
    closeAutomaticReview();
    automaticRevision = state.status.gallery_revision;
    automaticReview = createReferenceReview({ ...item, revision: automaticRevision }, async (reference, index) => {
      if (await actions.reviewAutomatic(reference, index)) {
        closeAutomaticReview();
        $("identity-search").focus();
      }
    }, () => {
      closeAutomaticReview();
      $("identity-search").focus();
    }, actions.refresh);
    $("automatic-item").replaceChildren(automaticReview.element);
    const section = $("automatic-section");
    section.hidden = false;
    section.tabIndex = -1;
    section.focus();
  }

  function renderPending(item) {
    review?.dispose();
    review = item ? createReferenceReview(item, actions.review, null, actions.refresh) : null;
    $("pending-item").replaceChildren(...(review ? [review.element] : []));
    updateReviewControls();
  }

  function openPendingReview(id) {
    const item = state.status?.pending.find(reference => reference.pending_id === id);
    if (!item) return;
    if (id !== activePendingId) {
      activePendingId = id;
      renderPending(item);
      for (const card of $("identity-grid").querySelectorAll("button[data-pending-id]")) {
        card.setAttribute("aria-pressed", String(card.dataset.pendingId === id));
      }
      updatePendingPosition(state.status.pending);
    }
    focusPendingReview();
  }

  function focusPendingReview() {
    const section = $("pending-section");
    section.tabIndex = -1;
    section.focus();
  }

  function updatePendingPosition(pending) {
    const index = pending.findIndex(item => item.pending_id === activePendingId);
    $("pending-position").textContent = index < 0 ? "" : `Reference ${index + 1} of ${pending.length} pending`;
  }

  function render() {
    const { status, operation, galleryFiles, galleryUpload } = state;
    if (automaticReview && automaticRevision !== status?.gallery_revision) closeAutomaticReview();
    const pending = status?.pending ?? [];
    const pipeline = status?.active_pipeline_id;
    const samePipeline = pipeline === activePipeline;
    const previousId = activePendingId;
    const previousIndex = previousPending.findIndex(item => item.pending_id === previousId);
    const reviewed = samePipeline && operation?.kind === "review" && previousIndex >= 0
      && !pending.some(item => item.pending_id === previousId);
    const nextIndex = reviewed && pending.length ? previousIndex % pending.length : 0;
    const next = (samePipeline && pending.find(item => item.pending_id === previousId))
      || pending[nextIndex];
    activePendingId = next?.pending_id;
    activePipeline = pipeline;
    previousPending = pending;
    $("pending-section").hidden = !pending.length;
    if (!samePipeline || previousId !== activePendingId) {
      renderPending(next);
      if (reviewed && state.workspace === "gallery") {
        if (next) focusPendingReview();
        else $("identity-search").focus();
      }
    }
    updatePendingPosition(pending);
    const canEnroll = state.connected && status?.engine_initialized && !operation;
    $("choose-folder").disabled = $("choose-references").disabled = !canEnroll;
    $("upload-gallery").disabled = !canEnroll || !galleryFiles.length;
    $("upload-gallery").textContent = operation?.kind === "gallery" ? "Enrolling…"
      : status?.gallery_loaded ? "Replace gallery" : "Build gallery";
    $("clear-gallery").disabled = !state.connected || Boolean(operation) || !(status?.gallery_loaded || pending.length);
    $("selected-files").textContent = galleryFiles.length ? `${galleryFiles.length} selected · ${galleryFiles.slice(0, 3).map(file => file.name).join(", ")}${galleryFiles.length > 3 ? ", …" : ""}` : "";
    $("gallery-count").textContent = status ? `${status.people_count} enrolled` : "Gallery unavailable";
    $("gallery-message").classList.toggle("enrollment-progress", operation?.kind === "gallery");
    if (status !== previousStatus || galleryUpload !== previousUpload || !rows.length) {
      rows = identityRows(state);
      renderRows();
      const counts = $("gallery-counts");
      counts.replaceChildren();
      if (status) {
        const rejected = rows.filter(item => item.status === "rejected").length;
        counts.append(el("span", `${status.people_count} accepted`, "tag green"),
          el("span", `${rejected} rejected`, "tag"), el("span", `${pending.length} pending`, "tag amber"));
        if (galleryUpload) {
          const automaticCount = galleryUpload.automatic_selection_references ?? 0;
          counts.append(el("small", `Last upload: ${galleryUpload.accepted_references} accepted (${automaticCount} automatically selected), ${galleryUpload.rejected_references} rejected, ${galleryUpload.manual_selection_references} sent to review.`));
        }
      }
      previousStatus = status;
      previousUpload = galleryUpload;
    }

    updateReviewControls();
  }

  for (const [button, input] of [["choose-folder", "gallery-folder"], ["choose-references", "gallery-files"]]) {
    $(button).addEventListener("click", () => $(input).click());
    $(input).addEventListener("change", event => {
      actions.choose(Array.from(event.target.files));
      event.target.value = "";
    });
  }
  $("identity-search").addEventListener("input", renderRows);
  $("identity-filter").addEventListener("change", renderRows);
  $("upload-gallery").addEventListener("click", actions.upload);
  $("clear-gallery").addEventListener("click", actions.clear);
  return render;
}
