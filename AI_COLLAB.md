# AI collaboration and engineering decisions

The project owner directed an offline open-set identification demo, chose what to evaluate and set the final submission requirements. AI assistance supported implementation, experiment design, debugging and documentation. The record distinguishes a decision visible in the project conversation from a conclusion reconstructed from source and saved experiments.

Codex's work is directly supported by the current project conversation, source, tests and verification reports. The owner reports using ChatGPT for earlier project discussion and Claude for an independent ZIP-based review and a documentation blueprint. The available source ZIP is consistent with a review snapshot, but the original Claude and ChatGPT conversations and the blueprint text were not located. Historical assistant wording and individual recommendations are therefore left unattributed unless directly observed.

## Decision record

| Decision | Context and AI input | Owner direction and reasoning | Outcome and supporting evidence |
|---|---|---|---|
| 1. Local browser architecture | The earlier app needed local inference and a usable group-photo workflow. The initial framework discussion is unavailable. | The owner explicitly required FastAPI and modular vanilla JavaScript, with no frontend build stack during the desktop milestone. That supported a local, inspectable deployment. | Codex refined the persistent shell and workspaces. [Architecture](ARCHITECTURE.md). A specific Streamlit recommendation could not be verified. |
| 2. Dataset evidence and scope | Detection and recognition require different labels and denominators. The later owner brief called for cautious dataset and licensing attribution. | Keep engineering decisions tied to the experiments actually run and limit the public record to aggregate results. | WIDER FACE supports box evaluation and LFW supports identity experiments. No original record establishes the reported SoloFace2 circularity assessment or use of volunteer or Wikimedia images. [Research](RESEARCH.md#7-dataset-and-rights-decisions). |
| 3. Resolution and size sensitivity | A local scaling utility shows that resize behavior was investigated. The exact assistant recommendation and historical image output were unavailable. | Preserve verified geometry and distinguish an observed association from a causal claim. | The current detector downsizes large frames and restores coordinates. The saved comparison quantifies recall by face size. The original before/after image counts remain open. [Detector](face_engine/detector.py), [research](RESEARCH.md#resolution-sensitivity). |
| 4. Duplicate identities | Codex added a post-match conflict pass, documented in the earlier collaboration record. Independently matched faces can compete for one enrolled identity. | Preserve duplicate protection and do not publish private identities or the unreproducible example. | Highest-score retention, deterministic tie-breaking, and model-specific demotion have synthetic regressions. [Conflict tests](tests/test_pipeline_conflicts.py). |
| 5. Detector choice and experiment scope | The documented evidence includes YuNet, SCRFD, RetinaFace, and a separate partial YOLO comparison. Individual historical AI recommendations are unavailable. | Use saved evidence, distinguish cohorts and preserve the interrupted run without rerunning expensive work for documentation. | SCRFD yielded stronger recall and F1 on the completed three-way sample. The YOLO comparison uses only the shared 1,500 images. [Detector results](RESEARCH.md#2-detector-experiments). |
| 6. Repair the open-set cohort | An early run evaluated known probes for identities absent from the built gallery. The artifact records the defect. The origin of the first diagnosis is not independently attributable. | The final brief requires the invalid cohort and corrected methodology to remain visible. | Version 2 preflighted references, used reserves and checked enrolled/probe consistency. Counts changed, so the reports are kept separate. [Open-set discussion](RESEARCH.md#pair-verification-and-corrected-open-set-identification). |
| 7. Select a CPU recognizer | The recognition comparison evaluated three models using shared YuNet detections. The original recommendation transcript is unavailable. | The owner chose AdaFace for the preferred research pipeline and retained SFace for fallback. CPU cost and model size mattered alongside accuracy. | AdaFace matched ArcFace's correct known acceptance on the fixed cohort at lower CPU cost. The threshold deployed later came from the independently calibrated combined run. [Recognition evidence](RESEARCH.md#3-recognition-investigation). |
| 8. Integrate SCRFD geometry | The runtime has explicit stride padding and a regression test. The historical console trace is absent. | Preserve the preferred detector and verify original-coordinate handling. | Aspect-preserving resize, bottom/right padding, and inverse coordinate scaling are in the adapter. [Implementation](face_engine/research_models.py), [regression](tests/test_research_models.py). |
| 9. Respond to the independent review | The owner reports that Claude raised fresh-clone startup, rights, documentation and reproducibility issues and recommended reconsidering the default. The review response itself is unavailable. | The owner explicitly directed that the preferred research pipeline remain SCRFD + AdaFace, with graceful fallback addressing startup reliability. | Startup tries the alternate bundle and reports failure reasons. A failed interactive switch preserves the active state. [Configuration](face_engine/configuration.py), [verification](docs/verification/MILESTONES.md). |
| 10. Use one Python environment and verify recovery | Codex observed a broken interpreter path and used a temporary official Python 3.11 runtime with the checkout's installed packages. The dependency record separated the declared OpenCV package name from the usable `cv2` provider. | The owner required reproducible setup and autonomous functional testing within execution permissions. | The combined pins and documented InsightFace installation preserve one OpenCV installation. Tests and isolated HTTP exercises covered failed enrollment, switching and cache restoration. [Setup](README.md#installation-and-first-run). |
| 11. Define the desktop product | Codex mapped existing browser actions and API fields before implementing Dashboard, Gallery, Analyze and Settings. | The owner specified the workspaces, desktop target sizes, truthful live statistics, local exports and pipeline choice. The later request constrained changes to visual refinement. | The shared state and local modules retain analytics provenance, gallery/thumb revisions and analysis invalidation. Browser access failed, so verification relies on HTTP and regression checks. [Verification](docs/verification/MILESTONES.md). |
| 12. Satisfy the artifact and evidence requirements | The assignment calls for genuine Archify output and traceable collaboration. Codex located saved experiments and used the official renderer. | The owner required authentic excerpts only, a private claim inventory, no raw biometric data and no commit or push. | The generated diagram and sanitized summaries are present. Original ChatGPT/Claude exports remain outstanding. [Architecture](ARCHITECTURE.md), [evidence](docs/research/EVIDENCE.md). |
| 13. Investigate the SCRFD gallery misses | The owner observed that YuNet ultimately enrolled all 17 references while SCRFD rejected three, and asked Codex to trace the exact failures. The controlled runs reproduced the difference and recovered all three faces at a smaller scale with the same SCRFD threshold. | The owner specified a gallery-only retry at 640 pixels after an empty primary pass, preserving the selected detector and existing matching thresholds. | The retry maps boxes and landmarks back to the working image for AdaFace. The additional validation uses separate photos and controls. [Research](RESEARCH.md#reference-enrollment-scale-sensitivity), [architecture](ARCHITECTURE.md#enrollment). |
| 14. Inspect the extra candidate | The owner noticed a reference with one visible face that triggered manual review. Codex inspected both production crops and located the second detection on a brick recess. | Keep SCRFD and the existing thresholds, and make manual review easier to inspect. | Candidate crops, raw detector scores, zoom, and synchronized numbered overlays expose the evidence before confirmation. [Architecture](ARCHITECTURE.md#enrollment). |
| 15. Reduce demo review while retaining control | The owner requested simpler dominant-face selection for the demonstration, as recorded in the release instructions. Saved validation shows that the heuristic picked the larger person in three cases where the intended target was the smaller person. | Keep strict review selectable and let the user inspect or replace an automatic choice. | Smart selection reduced required reviews from three to one in the 17-reference folder. A correction replaces the vector and advances the revision. The smaller-target failures remain documented. [Research](RESEARCH.md#smart-reference-auto-selection). |

## Verbatim excerpts from the available Codex conversation

The following are complete individual sentences from the conversation available in this workspace session. Surrounding message text is omitted. Milestone labels identify the source exchange; no share URL or export identifier was available.

**Codex, Milestone 3 verification.** Connects to decisions 9–11.

> User: “If browser automation remains unavailable, perform HTTP-level integration tests and provide a precise manual UI checklist.”

> Codex: “The browser host is still unavailable. I found an existing local 50-reference fixture and will use it against an isolated FastAPI instance to verify enrollment, analysis, and pipeline cache restoration.”

**Codex, Milestone 4 kickoff.** Connects to decisions 9 and 12.

> User: “Do not revert the preferred pipeline to YuNet + SFace merely because an earlier review suggested doing so.”

> Codex: “I’ll also read the official Archify instructions and check what the tool processes before running it.”

These are excerpts from the active conversation. They do not constitute a full export or independent evidence of the original Claude review.

**Codex, adaptive reference-enrollment follow-up.** Connects to decision 13.

> User: “Do not assume that SCRFD is inherently superior because of the WIDER FACE benchmark.”

> Codex: “Try a gallery-only SCRFD pass at 640 pixels when the standard pass returns zero faces.”

## Full-history evidence still needed

- TODO — Supply the original Claude review of the FaceFind ZIP and the follow-up blueprint, ideally with date or original conversation identifier.
- TODO — Supply the original ChatGPT research and architecture conversation, including any framework or dataset trade-offs and the cited SoloFace2 dataset-card revision.
- TODO — Export the Codex milestone conversation if the assignment requires the full history. The current excerpts provide only selected decisions.

No complete conversation export was found in the repository, inspected review archive, or checked project-related artifact locations. Keep originals private and review a separate submission copy for credentials, personal contact information, filesystem paths, biometrics, identities and unrelated content. The archive is not a substitute for review commentary.

## Reflection

The owner set the problem, selected the development and comparison scope, directed the final pipeline choice and set the standard for evidence and privacy. Codex handled source analysis, implementation and verification within that scope. The clearest value of the experiments was exposing denominator errors and measuring model trade-offs before choosing a demo default. The reported independent review drew attention to the startup experience. Fallback and status reporting addressed that concern while preserving the intended research configuration. The incomplete browser and conversation evidence marks where human follow-through is still needed.

## Final independent review follow-up

The human-directed hardening request identified three concrete release issues: affected upload dependencies, stale review actions that could target a different photo with the same identity stem, and distinct Unicode names that could render identically. The implementation updates the upload stack against published advisories, separates review-item identity from embedding revision, and uses local font rendering with an explicit face-number fallback. The verification report records the regressions and the clean-install checks. The original review conversation export still needs to be supplied and sanitized as described above. [Release verification](docs/verification/RELEASE.md).
