# FaceFind engineering milestones

For the latest counts and isolated startup and workflow checks, see the [release report](RELEASE.md) and the [assignment compliance matrix](COMPLIANCE.md). The results below describe the earlier milestones.

This is a sanitized record of functional checks and documentation validation. Identity labels and images are excluded.

| Milestone | Evidence and verified scope | Limitation |
|---|---|---|
| 1 · startup and selection | A separately labeled Milestone 1 report was unavailable. Current startup regressions cover preferred selection, unavailable dependencies and checkpoints, fallback, both engines unavailable, failed switch and compatible cache restoration. Milestone 4 also exercised both explicit bundled startup and automatic fallback over HTTP from a temporary source copy containing only bundled model files. | Earlier Milestone 1 execution details remain unavailable. |
| 2 · recognition workflow | Saved `facefind_milestone2_verification_20261124_01` report. Actual SCRFD + AdaFace API analyses returned 14 detections twice. A five-reference smoke test reported one direct enrollment, three rejections and one pending face selection that raised enrollment to two. The repeated analysis instrumentation counted 28 per-face embedding calls and zero gallery embedding calls. The adapter post-check produced a normalized 512-dimensional vector. | The genuine group photo has no verified identities. Smoke timings are not benchmark results. |
| 3 · desktop implementation | Isolated HTTP verification enrolled 50 references, decoded 50 bounded thumbnails, exercised six rejection reasons, manually confirmed a pending face, rejected stale requests and kept the working gallery after a failed replacement. Repeat group analysis returned 11 detections, one recognized and ten unknown, with no assigned identity for the unknown rows. The annotated JPEG decoded at the source dimensions and gallery embeddings were reused. Both caches restored after switching and a missing-checkpoint switch returned 503 without changing the fallback gallery. | These are HTTP observations, not browser interactions or identity-accuracy measurements. |
| 3 · visual refinement | Static module delivery, 57 direct DOM bindings, unique IDs, Python regressions and JavaScript logic verified. CSV tests exercise formula escaping, quoting and unknown-identity suppression. | The authorized browser host was unavailable. These remained open at Milestone 3. The final release later added targeted Chrome verification. |
| 4 · documentation | Numeric tables checked against saved outputs, custom detection counts reconciled with precision/recall/F1, model files hashed locally, and Archify generation validated. No model benchmark rerun. | The Archify viewer's automated browser gate passed. It did not exercise the FaceFind UI. |

## Published development milestones

| Committed milestone | Contribution | Verification and scope |
|---|---|---|
| 21 Nov 2026 · `8694ec4` | Initial recognition engine and project foundation. | Earliest committed implementation. The initiating prompt has not been recovered. |
| 22–23 Nov 2026 · `aab95a7`, `9661ea9` | Offline browser application and modular recognition pipeline. | Git records the application foundation and modularization. |
| 24 Nov 2026 · `819f40a`, `2cdcc1f` | SCRFD and AdaFace integration and local recognition release, with bundled fallback. | Preferred and bundled startup exercised in isolated runtimes. |
| 24 Nov 2026 · `9734b20` | Independent ground-truth Evaluation Studio and frozen application release. | Final Chrome and clean-clone checks, 103 Python passes with four skips, and 14 JavaScript passes. |
| 24 Nov 2026 · `c42b122`, `5a524db` | Disconnected-network acceptance evidence and interview presentation. | Developer offline acceptance, plus an inspected 11-slide PowerPoint and PDF. |

## Earlier regression tests · November 24, 2026

~~~powershell
python -m pytest -q tests scripts/evaluation -rs
node --experimental-default-type=module --test tests/test_desktop_frontend.mjs
~~~

- Python 3.11.9 returned **73 passed, 2 skipped**. The skips in the evaluation utility suites require Windows directory-symlink privileges.
- Node 22.17.1 returned **4 passed**. All JavaScript modules passed syntax checks.
- Bundled and automatic-fallback HTTP startup checks passed against an isolated source copy. A separate temporary installation of the combined requirements and InsightFace 2.1 matched every direct pin, initialized SCRFD + AdaFace on CPU, and reran the Python suite with the same result. The temporary Python runtime is distinct from the PowerShell `py` launcher in the setup guide. The host's existing launcher path was unavailable, so Windows activation itself was not retested.
- Documentation validation checked 57 local links and anchors, heading and table structure, the generated viewer's resource tags, public-file privacy patterns, and whitespace with `git diff --check`. Numeric cross-checks covered the main detector, recognition, combined and application tables as well as seven local checkpoint sizes and SHA-256 hashes.
- Archify 3.3.0 passed `doctor`, schema validation, generation, strict artifact checking and automated browser checking on the successful retry. See [the diagram and procedure](../../ARCHITECTURE.md#archify-artifact).

## Later browser coverage

The final release records targeted Chrome enrollment, Analyze, Evaluation Studio, download and stale-preview recovery. The developer recording independently documents the manual native-folder interaction and its resulting 24-reference enrollment. These observations do not imply a comprehensive cross-platform accessibility pass. [Final browser verification](RELEASE.md#evaluation-studio-verification) · [Optional validation](MISSING-EVIDENCE.md#optional-future-validation)
