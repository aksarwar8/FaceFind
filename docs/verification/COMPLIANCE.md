# Assignment compliance matrix

The original assignment brief was unavailable in the checked local project files and likely project-document locations. This matrix covers the requirements supplied for Milestone 5. PASS means the listed scope was verified, PARTIAL means evidence is incomplete, FAIL means a requirement was demonstrably unmet, and NOT VERIFIED means no direct verification was available. See the [release verification report](RELEASE.md).

| Requirement | Implementation | Supporting file | Evidence | Status |
|---|---|---|---|---|
| Reference-folder selection | Browser directory input and multipart gallery | [HTML](../../web/index.html), [API](../../api/routes.py) | HTTP upload passed. Native folder picker remains manual. | PARTIAL |
| Approximately 50 named references | Filename-based gallery with optional review and correction | [Gallery API](../../api/routes.py) | 52 final enrollments in isolated preferred run. | PASS |
| Group-photo analysis | Local upload, preview and recognition | [Analysis API](../../api/routes.py) | Genuine group photo analyzed in both environments. | PASS |
| Face detection | SCRFD and YuNet adapters | [Research models](../../face_engine/research_models.py), [YuNet](../../face_engine/detector.py) | Both real detectors ran; diagnostics and control cases saved. | PASS |
| Known-person identification | Gallery cosine matching and margin | [Matcher](../../face_engine/match.py), [research](../../RESEARCH.md) | Labeled historical comparisons and assigned labels in the current unannotated fixture. | PASS |
| Unknown-person handling | Explicit Unknown without confirmed identity | [Pipeline](../../face_engine/pipeline.py) | Six Unknown rows in the preferred demo, all with null assigned identities. | PASS |
| Bounding boxes and labels | Working-image coordinates and annotated JPEG | [API](../../api/routes.py) | Structured boxes and decoded output checked. Alignment and interactive highlighting remain manual in the browser. | PARTIAL |
| Offline operation | Same-origin frontend and CPU inference | [Architecture](../../ARCHITECTURE.md) | Process-level Python network guard active during inference. Full browser and disconnected-network checks pending. | PARTIAL |
| User interface | Dashboard, Gallery, Analyze and Settings | [Frontend](../../web/) | Local assets and JavaScript tests passed. Rendered layout and browser interactions pending. | PARTIAL |
| README | Evaluator setup and workflow | [README](../../README.md) | Setup path exercised. Safe screenshots pending. | PASS |
| Research | Protocols, trade-offs, evidence and limitations | [Research](../../RESEARCH.md) | Saved aggregates and provenance cross-checked, with gaps identified. | PASS |
| Architecture | Components, lifecycle, contracts and privacy | [Architecture](../../ARCHITECTURE.md) | Checked against the release source. | PASS |
| Genuine Archify | Standalone HTML and editable specification | [Archify diagram](../architecture/facefind.html) | Archify 3.3.0 validation, delivery, strict check and automated viewer check passed. | PASS |
| AI collaboration history | Decision log and authenticated excerpts from the available conversation | [AI collaboration](../../AI_COLLAB.md) | Original Claude and ChatGPT exports remain unavailable. | PARTIAL |
| Public repository reproducibility | Pinned Windows Python 3.11 base and optional setup | [Requirements](../../requirements.txt), [README](../../README.md) | Clean allowlisted source snapshot installed and ran. The final public Git revision has not been published or re-cloned. | PARTIAL |
| Full official brief review | Reconcile final deliverables with the original brief | [Release assessment](RELEASE.md#outstanding-issues) | Original brief unavailable. | NOT VERIFIED |
## Hardening verification

The release verification report covers the updated dependency pins, immutable review actions and Unicode JPEG labels. The assignment brief and authentic conversation exports remain separate submission evidence. [Release verification](RELEASE.md).
