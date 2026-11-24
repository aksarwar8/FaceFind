# FaceFind verification record

For the latest counts and isolated startup and workflow checks, see the [Milestone 5 release report](RELEASE.md) and the [assignment compliance matrix](COMPLIANCE.md). The results below describe the earlier milestones.

This is a sanitized record of functional checks and documentation validation. Identity labels and images are excluded.

| Milestone | Evidence and verified scope | Limitation |
|---|---|---|
| 1 · startup and selection | A separately labeled Milestone 1 report was unavailable. Current startup regressions cover preferred selection, unavailable dependencies and checkpoints, fallback, both engines unavailable, failed switch and compatible cache restoration. Milestone 4 also exercised both explicit bundled startup and automatic fallback over HTTP from a temporary source copy containing only bundled model files. | Earlier Milestone 1 execution details remain unavailable. |
| 2 · recognition workflow | Saved `facefind_milestone2_verification_20261124_01` report. Actual SCRFD + AdaFace API analyses returned 14 detections twice. A five-reference smoke test reported one direct enrollment, three rejections and one pending face selection that raised enrollment to two. The repeated analysis instrumentation counted 28 per-face embedding calls and zero gallery embedding calls. The adapter post-check produced a normalized 512-dimensional vector. | The genuine group photo has no verified identities. Smoke timings are not benchmark results. |
| 3 · desktop implementation | Isolated HTTP verification enrolled 50 references, decoded 50 bounded thumbnails, exercised six rejection reasons, manually confirmed a pending face, rejected stale requests and kept the working gallery after a failed replacement. Repeat group analysis returned 11 detections, one recognized and ten unknown, with no assigned identity for the unknown rows. The annotated JPEG decoded at the source dimensions and gallery embeddings were reused. Both caches restored after switching and a missing-checkpoint switch returned 503 without changing the fallback gallery. | These are HTTP observations, not browser interactions or identity-accuracy measurements. |
| 3 · visual refinement | Static module delivery, 57 direct DOM bindings, unique IDs, Python regressions and JavaScript logic verified. CSV tests exercise formula escaping, quoting and unknown-identity suppression. | The authorized browser host was unavailable. App screenshots, viewport layout, browser file/download workflows and network trace remain open. |
| 4 · documentation | Numeric tables checked against saved outputs, custom detection counts reconciled with precision/recall/F1, model files hashed locally, and Archify generation validated. No model benchmark rerun. | The Archify viewer's automated browser gate passed. It did not exercise the FaceFind UI. |

## Latest regression tests · November 24, 2026

~~~powershell
python -m pytest -q tests scripts/evaluation -rs
node --experimental-default-type=module --test tests/test_desktop_frontend.mjs
~~~

- Python 3.11.9 returned **73 passed, 2 skipped**. The skips in the evaluation utility suites require Windows directory-symlink privileges.
- Node 22.17.1 returned **4 passed**. All JavaScript modules passed syntax checks.
- Bundled and automatic-fallback HTTP startup checks passed against an isolated source copy. A separate temporary installation of the combined requirements and InsightFace 2.1 matched every direct pin, initialized SCRFD + AdaFace on CPU, and reran the Python suite with the same result. The temporary Python runtime is distinct from the PowerShell `py` launcher in the setup guide. The host's existing launcher path was unavailable, so Windows activation itself was not retested.
- Documentation validation checked 57 local links and anchors, heading and table structure, the generated viewer's resource tags, public-file privacy patterns, and whitespace with `git diff --check`. Numeric cross-checks covered the main detector, recognition, combined and application tables as well as seven local checkpoint sizes and SHA-256 hashes.
- Archify 3.3.0 passed `doctor`, schema validation, generation, strict artifact checking and automated browser checking on the successful retry. See [the diagram and procedure](../../ARCHITECTURE.md#archify-artifact).

## Manual browser checks still required

1. At 1366 × 768 and 1920 × 1080, inspect the Dashboard, Gallery Manager, Analyze and Settings. Confirm keyboard access, focus, navigation state retention and overflow behavior.
2. Choose a folder of roughly 50 safe reference images. Verify enrollment feedback, portraits, search, filtering, pending-face selection, and rejection details after failed replacement.
3. Select and drag in a group photo. Verify source dimensions, original/annotated toggle, zoom and reset, bounding-box highlighting, result panel scrolling and downloads. Confirm unknown rows do not display an assigned identity.
4. Switch each way in Settings. Confirm actual active state, cached gallery restoration, invalidation of old results, and fallback or failure feedback. Recheck connectivity after a local backend stop/restart.
5. Inspect the browser Network panel for same-origin application requests. Capture four publishable screenshots with authorized inputs and add them to the README after review.
