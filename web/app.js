
const API = "/api";

const $ = (id) => document.getElementById(id);

let selectedGalleryFiles = [];
let selectedGroupFile = null;
const SUPPORTED_GALLERY_EXTENSIONS = new Set([
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
]);

async function apiRequest(endpoint, options = {}) {
    const response = await fetch(`${API}${endpoint}`, options);

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(data.detail || "Something went wrong.");
    }

    return data;
}

function showMessage(element, message, type = "") {
    element.textContent = message;
    element.className = `message ${type}`;
}

function updateGalleryUI(data) {
    $("people-count").textContent = data.people_count ?? 0;
    $("pending-count").textContent = data.pending_references ?? 0;

    $("gallery-badge").textContent =
        data.gallery_loaded ? "READY" : "EMPTY";

    $("gallery-summary").textContent =
        data.gallery_loaded
            ? `${data.people_count} identities loaded successfully.`
            : "Your reference gallery is empty.";

    $("clear-gallery").disabled = !data.gallery_loaded &&
        !data.pending_references;

    $("analyze-photo").disabled =
        !data.gallery_loaded || !selectedGroupFile;

    const pending = data.pending || [];
    $("pending-section").hidden = pending.length === 0;

    renderPending(pending);
    if (data.active_pipeline_id && $("pipeline-select")) {
        $("pipeline-select").value = data.active_pipeline_id;
    }
}

async function refreshStatus() {
    try {
        const [health, status] = await Promise.all([
            apiRequest("/health"),
            apiRequest("/status")
        ]);

        $("engine-status").textContent =
            health.engine_initialized ? "READY" : "OFFLINE";

        updateGalleryUI(status);
    } catch (error) {
        $("engine-status").textContent = "ERROR";
        showMessage($("gallery-message"), error.message, "error");
    }
}

$("pipeline-select").addEventListener("change", async (event) => {
    const select = event.target;
    select.disabled = true;
    showMessage($("pipeline-message"), "Loading local models...");
    try {
        const data = await apiRequest("/pipeline", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pipeline_id: select.value })
        });
        updateGalleryUI(data);
        showMessage(
            $("pipeline-message"),
            `${data.active_pipeline_label} ready (${data.model_initialization_ms.toFixed(1)} ms initialization). ` +
            (data.gallery_loaded ? "Its cached gallery is ready." : "Build a gallery for this pipeline."),
            "success"
        );
    } catch (error) {
        showMessage($("pipeline-message"), error.message, "error");
        await refreshStatus();
    } finally {
        select.disabled = false;
    }
});


// Reference image selection
function validateGalleryFiles(files) {
    const supported = files.filter(file =>
        SUPPORTED_GALLERY_EXTENSIONS.has(
            file.name.slice(file.name.lastIndexOf(".")).toLowerCase()
        )
    );
    const unsupported = files.filter(file => !supported.includes(file));
    const identityFiles = new Map();
    const duplicateIdentities = new Set();

    for (const file of supported) {
        const identity = file.name.slice(0, file.name.lastIndexOf(".")).trim();
        const key = identity.toLocaleLowerCase();

        if (identityFiles.has(key)) {
            duplicateIdentities.add(identityFiles.get(key));
            duplicateIdentities.add(identity);
        } else {
            identityFiles.set(key, identity);
        }
    }

    return { supported, unsupported, duplicateIdentities };
}

function updateSelectedGalleryFiles(files) {
    selectedGalleryFiles = files;
    const validation = validateGalleryFiles(files);
    const countLabel = `${validation.supported.length} supported image${
        validation.supported.length === 1 ? "" : "s"
    } selected`;
    const names = files.map(file => file.name).join(" · ");

    $("gallery-file-list").textContent = files.length
        ? `${countLabel}${names ? ` · ${names}` : ""}`
        : "";

    const messages = [];
    if (validation.unsupported.length) {
        messages.push(
            `Unsupported files will be skipped: ${validation.unsupported
                .map(file => file.name).join(", ")}`
        );
    }
    if (validation.duplicateIdentities.size) {
        messages.push(
            `Duplicate identity names: ${Array.from(
                validation.duplicateIdentities
            ).join(", ")}. Remove or rename duplicates before uploading.`
        );
    }

    showMessage(
        $("gallery-message"),
        messages.join("\n"),
        messages.length ? "error" : ""
    );
    $("upload-gallery").disabled =
        validation.supported.length === 0 ||
        validation.duplicateIdentities.size > 0;
}

$("gallery-files").addEventListener("change", (event) => {
    updateSelectedGalleryFiles(Array.from(event.target.files || []));
});

$("gallery-folder").addEventListener("change", (event) => {
    updateSelectedGalleryFiles(Array.from(event.target.files || []));
});


$("upload-gallery").addEventListener("click", async () => {
    if (!selectedGalleryFiles.length) return;

    const validation = validateGalleryFiles(selectedGalleryFiles);
    if (!validation.supported.length || validation.duplicateIdentities.size) return;

    const button = $("upload-gallery");
    button.disabled = true;
    button.textContent = "Processing photographs...";

    try {
        const formData = new FormData();

        validation.supported.forEach(file => {
            formData.append("files", file);
        });

        const data = await apiRequest("/gallery", {
            method: "POST",
            body: formData
        });

        updateGalleryUI(data);

        let message = data.message;

        if (validation.unsupported.length) {
            message += "\nSkipped unsupported files: " +
                validation.unsupported.map(file => file.name).join(", ");
        }

        if (data.gallery_errors?.length) {
            message += "\n" + data.gallery_errors.join("\n");
        }

        const rejectedBuild = data.build_succeeded === false;
        if (data.upload_summary) {
            message += `\nAccepted ${data.upload_summary.accepted_references}; rejected ${data.upload_summary.rejected_references}; manual selection ${data.upload_summary.manual_selection_references}.`;
        }
        showMessage($("gallery-message"), message, rejectedBuild ? "error" : "success");

        selectedGalleryFiles = [];
        $("gallery-files").value = "";
        $("gallery-folder").value = "";
        $("gallery-file-list").textContent = "";

    } catch (error) {
        showMessage($("gallery-message"), error.message, "error");
    } finally {
        button.disabled = false;
        button.textContent = "Build reference gallery";
    }
});


// Manual reference selection
function renderPending(pending) {
    const container = $("pending-list");
    container.replaceChildren();

    pending.forEach(item => {
        const card = document.createElement("div");
        card.className = "pending-card";

        const title = document.createElement("strong");
        title.textContent = item.identity;

        const details = document.createElement("p");
        details.textContent =
            `${item.filename} · ${item.faces} faces detected`;

        const preview = document.createElement("img");
        preview.alt = `Numbered face preview for ${item.identity}`;
        preview.loading = "lazy";

        const actions = document.createElement("div");
        actions.className = "pending-actions";

        for (let i = 0; i < item.faces; i++) {
            const button = document.createElement("button");
            button.textContent = `Select face ${i}`;
            button.addEventListener("click", () => {
                selectPendingFace(item.pending_id, i);
            });
            actions.appendChild(button);
        }

        const skip = document.createElement("button");
        skip.textContent = "Skip image";
        skip.addEventListener("click", () => {
            skipPending(item.pending_id);
        });
        actions.appendChild(skip);

        card.append(title, details, preview, actions);
        container.appendChild(card);

        apiRequest(`/gallery/pending/${item.pending_id}`)
            .then(data => {
                preview.src = data.preview;
            })
            .catch(error => {
                details.textContent += ` · ${error.message}`;
            });
    });
}


async function selectPendingFace(pendingId, faceIndex) {
    try {
        const data = await apiRequest("/gallery/select", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                pending_id: pendingId,
                face_index: faceIndex
            })
        });

        updateGalleryUI(data);

        showMessage(
            $("gallery-message"),
            data.message,
            "success"
        );
    } catch (error) {
        showMessage($("gallery-message"), error.message, "error");
    }
}


async function skipPending(pendingId) {
    try {
        const data = await apiRequest("/gallery/skip", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pending_id: pendingId })
        });

        updateGalleryUI(data);

        showMessage(
            $("gallery-message"),
            data.message,
            "success"
        );
    } catch (error) {
        showMessage($("gallery-message"), error.message, "error");
    }
}


// Clear gallery
$("clear-gallery").addEventListener("click", async () => {
    if (!confirm("Are you sure you want to clear the reference gallery?")) {
        return;
    }

    try {
        const data = await apiRequest("/gallery/clear", {
            method: "POST"
        });

        selectedGroupFile = null;
        $("group-file").value = "";
        $("group-file-name").textContent = "";

        $("results-section").hidden = true;
        $("analyze-photo").disabled = true;

        updateGalleryUI(data);

        showMessage(
            $("gallery-message"),
            data.message,
            "success"
        );
    } catch (error) {
        showMessage($("gallery-message"), error.message, "error");
    }
});


// Group photo selection
$("group-file").addEventListener("change", (event) => {
    selectedGroupFile = event.target.files?.[0] || null;

    $("group-file-name").textContent =
        selectedGroupFile ? selectedGroupFile.name : "";

    $("analyze-photo").disabled =
        !selectedGroupFile ||
        Number($("people-count").textContent) === 0;
});


// Group photo recognition
$("analyze-photo").addEventListener("click", async () => {
    if (!selectedGroupFile) return;

    const button = $("analyze-photo");
    button.disabled = true;
    button.textContent = "Analysing photograph...";

    showMessage(
        $("analysis-message"),
        "Detecting and recognising faces..."
    );

    try {
        const formData = new FormData();
        formData.append("file", selectedGroupFile);

        const data = await apiRequest("/analyze", {
            method: "POST",
            body: formData
        });

        $("face-total").textContent =
            `${data.face_count} FACES`;

        $("recognized-count").textContent =
            data.summary.recognised;

        $("possible-count").textContent =
            data.summary.possible;

        $("unknown-count").textContent =
            data.summary.unknown;

        $("annotated-image").src = data.annotated_image;

        renderResults(data.results);

        $("results-section").hidden = false;

        showMessage(
            $("analysis-message"),
            data.message,
            "success"
        );

    } catch (error) {
        showMessage($("analysis-message"), error.message, "error");
    } finally {
        button.disabled = !selectedGroupFile ||
            Number($("people-count").textContent) === 0;

        button.textContent = "Analyse group photograph";
    }
});



function renderResults(results) {
    const container = $("result-list");
    container.replaceChildren();

    results.forEach(result => {
        const item = document.createElement("div");
        item.className = "result-item result-item-expanded";

        const information = document.createElement("div");
        information.className = "result-information";

        const identity = document.createElement("strong");
        identity.textContent = result.conflict_with
            ? `Unconfirmed duplicate: ${result.identity || "Unknown person"}`
            : result.identity || "Unknown person";

        const details = document.createElement("small");
        details.textContent =
            `Face ${result.face_number} · Best similarity: ${
                result.best_score != null
                    ? (result.best_score * 100).toFixed(1) + "%"
                    : "N/A"
            }`;
        if (result.conflict_with) {
            details.textContent +=
                ` · Same identity retained for face ${result.conflict_with}`;
        }

        information.append(identity, details);

        const status = document.createElement("span");
        status.className = `result-status ${result.status}`;
        status.textContent = result.status.toUpperCase();

        item.append(information, status);

        // Display alternative candidates when available.
        if (result.candidates?.length) {
            const candidates = document.createElement("div");
            candidates.className = "candidate-list";

            const heading = document.createElement("small");
            heading.textContent = "TOP MATCH CANDIDATES";
            candidates.appendChild(heading);

            result.candidates.forEach(candidate => {
                const row = document.createElement("div");
                row.className = "candidate-row";

                const name = document.createElement("span");
                name.textContent = candidate.name;

                const score = document.createElement("span");
                score.textContent =
                    `${(candidate.score * 100).toFixed(1)}%`;

                row.append(name, score);
                candidates.appendChild(row);
            });

            item.appendChild(candidates);
        }

        container.appendChild(item);
    });
}


// Initialize dashboard
refreshStatus();
apiRequest("/pipeline").then(data => {
    $("pipeline-select").value = data.active_pipeline_id;
    showMessage(
        $("pipeline-message"),
        `${data.active_pipeline_label} ready (${data.model_initialization_ms.toFixed(1)} ms initialization).`,
        "success"
    );
}).catch(error => showMessage($("pipeline-message"), error.message, "error"));
