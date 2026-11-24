# FaceFind release verification

**Frozen application** [`9734b203f524670b695e85b0ea2088f4c6794517`](https://github.com/aksarwar8/FaceFind/commit/9734b203f524670b695e85b0ea2088f4c6794517)
**Recorded environment** Windows x64, Python 3.11.9, Node 22.17.1, local CPU, November 24, 2026.

| Current release check | Verified result |
|---|---|
| Engine startup | Preferred SCRFD + AdaFace initialized with locally supplied checkpoints. The bundled-only public clone automatically fell back to YuNet + SFace. |
| Final regression suite | **103 Python passed, 4 skipped, 0 failed. 14 JavaScript passed.** The Python skips were two Windows directory-symlink checks and two Indic shaping checks. |
| Browser workflow | Chrome exercised upload, recognition, evaluation, and downloads. The preferred-pipeline synthetic cohort produced 56/57 correct outcomes across five composites with approximate boxes. |
| Disconnected-network acceptance | The developer confirmed startup and local operation while disconnected. The screenshot records the startup state. |
| Security and robustness | The checks covered local API boundaries, intake limits, cleanup, stale-reference responses, gallery revision changes and annotation fallback. |

## Public clone verification

### Evaluation Studio release

A fresh GitHub clone matched `9734b203f524670b695e85b0ea2088f4c6794517` and contained 91 tracked files. A separate Windows Python 3.11.9 runtime used the verified base packages only. It requested SCRFD + AdaFace and automatically fell back to bundled YuNet + SFace. Local Chrome enrolled five references from six uploads, analyzed a group image without ground truth, received an annotated JPEG and showed **Not Evaluated** when opening Evaluation Studio. The application returned 13 Unknown detections and zero processing errors. These are inference outcomes for the smoke fixture. All five evaluation modules were served locally, with no external browser requests or page errors recorded.

### Earlier bundled-model baseline

Public commit `2cdcc1fcacceb1abea083baf59ec1ccb2af5f078` was freshly cloned from GitHub. In an isolated Windows Python 3.11.9 runtime containing only the base requirements, explicit YuNet + SFace startup and automatic fallback both passed. Six references enrolled after one manual confirmation, and a group upload produced a decoded annotated JPEG without ground truth.

## Developer browser evidence

Dashboard, Gallery Manager, Analyze and the System Settings drawer show a ready local CPU engine, a 60-reference gallery and a completed group-photo analysis. These captures verify visible application states. [Screenshot register](../screenshots/README.md)

The Evaluation Studio capture comes from the saved Chrome run with imported ground truth. The five saved report hashes match their group images and all five reports are marked complete. The Easy capture shows 9/10 correct outcomes using approximate boxes and an IoU threshold of 0.50. The separate manual-correspondence screenshots lack a matching verified export and are excluded from benchmark evidence. [Controlled cohort](../../RESEARCH.md#evaluation-studio-demonstration)

### Native reference-folder enrollment

The private OBS recording shows the developer opening the operating-system folder picker (around 0:39), approving the folder upload in the browser (around 0:40), and reaching a processed gallery with 24 accepted references (around 1:18). This is developer-performed visual evidence of native folder selection. The separate isolated HTTP and Chrome runs exercised 52 and 60 references respectively. The recording and its photographs remain private.

## Developer-performed disconnected-network acceptance

The developer disconnected the laptop from the internet and confirmed that FaceFind started and worked correctly while disconnected. The supplied Dashboard capture shows **Backend connected**, **Engine ready**, and **SCRFD-10G-KPS + AdaFace IR-18**. It shows an empty gallery and no current analysis, so it documents the startup state. This manual acceptance check is separate from the earlier guarded runtime and fresh-clone Chrome network checks. [Redacted offline demo](../screenshots/offline-demo.png)

## Evaluation Studio verification

The Evaluation workspace was exercised in local Chrome with SCRFD + AdaFace, the private 60-photo gallery, and five synthetic demo composites. Browser checks covered the empty and Not Evaluated states, manual enrolled-name entry, partial progress, missed-box entry, both actual ground-truth formats, one-to-one results, synchronized selection and filtering, Dashboard status, and separate JSON, CSV and JPEG downloads. Opening another analysis cleared the previous evaluation. The exported prediction rows matched the original analysis response. The unannotated group images and private reports, including the browser screenshot, remain Git-ignored.

Archify 3.3.0 regenerated the architecture with the independent browser-side evaluation path. Validation, delivery, checks and the Archify browser gate passed. [Sanitized demo outcomes](../../RESEARCH.md#evaluation-studio-demonstration).

The published application keeps the existing recognition workflow usable without ground truth. Chrome also verified that three multi-face references were queued in Always Review mode, that a non-first card could be reviewed, and that a stale preview returned a recoverable HTTP 409.

The final suites returned 103 Python passes and four existing skips (two Windows directory-symlink checks and two Indic shaping checks), plus 14 JavaScript passes. A separate Chrome replay of the captured analysis verified the imported file SHA-256, boxless roster pairing, the incomplete-review guard and invalidation on a gallery revision change without a new inference run.

## Current scope and follow-up

| Category | Disposition |
|---|---|
| Resolved engineering findings | Same-origin and Host checks, bounded streamed multipart handling, immutable review IDs, rollover responsiveness, and safer annotation and error handling. See the historical baseline and hardening sections for the checked cases. |
| Current operational limitations | The deployment assumes one local workstation. HEIC/HEIF decoding was unavailable in the recorded environment. AVIF passed in the tested environment. Indic shaping and directory-symlink checks depend on host capabilities. |
| Outstanding assignment evidence | The available original conversations do not establish full history coverage. [Evidence register](MISSING-EVIDENCE.md#required-submission-evidence) |
| Optional future validation | Broader desktop and OS coverage and a representative annotated genuine group-photo cohort. [Research priorities](../../RESEARCH.md#future-research-and-validation) |

## Historical verification snapshots

The baseline and hardening records below document changes leading to the current release. Earlier counts and environment statements apply to their own runs.

## Baseline release-candidate audit and fixes

| Finding | Release change |
|---|---|
| Cross-origin form submissions could reach the unauthenticated localhost API. | Restrict the API to loopback Host headers and same-origin requests when Origin is present. Remove hosted Swagger and ReDoc assets; retain local OpenAPI JSON. |
| A truncated multipart body could leave an unfinished file outside normal form cleanup. Large advertised requests were rejected only as bytes arrived. | Require the closing boundary and close all spooled files on failure. Reject an oversized declared length before reading and retain streaming enforcement. Pin the Starlette version used by the parser. |
| Manual review could enroll a case variant of an existing name. A failed vector append could leave a name without a vector. | Check case-folded names in both insertion paths and append names after vector allocation succeeds. Reject invalid embedding norms. |
| Raw processing errors could expose local details. | Give the browser actionable validation or processing messages and keep internal details in local logs. Model status strips the checkout prefix. |
| Seven evaluation utility tests assumed a private output directory existed. | Create their fixtures under system temporary directories while preserving the assertions. |

The renderer inserts user content through DOM text APIs. The browser uses same-origin API calls. Static serving is confined to `web/`. Thumbnails take a revision and enrolled row index. The hardening update gives manual and automatic review items immutable IDs. No credential patterns matched the source-only review files. Gallery state and decoded review images stay in the process. The shared review budget for pending and auto-selected images is 256 MiB. Multipart uploads may spool to system temporary storage. The 100 MiB default per-image limit, 2 GiB per-gallery request limit, 128-file cap and 16 KiB part-header limit bound the intake. The decoder also limits source pixels and working-raster dimensions.

Live limit checks exercised the early aggregate 413, the 129-file rejection, and the multipart header limit. The gallery stayed empty after all three refused requests.

## Baseline installation and HTTP checks

| Scenario | Outcome |
|---|---|
| Evaluator source snapshot | Installed the base requirements into a clean temporary package directory, without ONNX Runtime, InsightFace, or experimental checkpoints. The default request for SCRFD + AdaFace fell back to bundled YuNet + SFace (128 dimensions). Six references enrolled after one pending confirmation. Analysis produced an annotated JPEG and detected 13 faces, with one recognized label and 12 Unknown. A failed replacement and failed switch preserved the working gallery. Foreign Origin and Host were rejected. |
| Preferred local demo | An isolated snapshot with the previously clean-installed combined pins, separately installed InsightFace 2.1, and two locally obtained research checkpoints started SCRFD + AdaFace. Both bundles initialized and each restored its own gallery after switches. An intentionally changed automatic selection advanced the revision and the old revision was rejected; the original choice was restored. Strict review was separately confirmed on the same reference. |
| Offline boundary | An audit hook in each isolated Uvicorn process blocked external Python DNS and TCP connection attempts. A self-test confirmed the block, and neither application made a blocked attempt during startup or inference. Browser asset URLs and served local modules were checked separately. This does not cover native-code network calls or constitute a disconnected-network browser demo. System networking was untouched. |
| Optional image codecs | The release environments verified JPEG, PNG, WebP, BMP and TIFF. HEIC/HEIF and AVIF decoders were absent. |
| Dependency caveat | InsightFace declares `opencv-python` in metadata. Both real pipelines ran with only `opencv-contrib-python` providing `cv2`. An absent-package warning for the declared provider is expected. Platform launcher activation was outside the temporary package test. |

## Approximately 50-reference demonstration

The preferred CPU run used 52 authorized reference files and one genuine group photo without verified identity annotations. The automated HTTP harness submitted deterministic choices to exercise the pending-confirmation route. These choices were for workflow verification.

| Item | Measured outcome |
|---|---:|
| Initial accepted / automatic subset / pending / rejected | 43 / 2 / 9 / 0 |
| Final enrolled | 52 |
| Gallery upload, before the nine confirmations | 11,959 ms |
| Detected / recognized label / unknown / error | 14 / 8 / 6 / 0 |
| First / repeated analysis request | 1,388 / 1,463 ms |
| First / repeated backend processing | 1,287 / 1,359 ms |

Repeat analysis reused the same gallery revision and cached embeddings. The JPEG decoded at the reported dimensions. The JavaScript CSV serializer processed the actual HTTP results, with one row per face and no assigned identities on unknown rows. This baseline exercised the serializer. The later Chrome run verified the download controls. [Final browser checks](#evaluation-studio-verification) Assigned labels are operational results. The fixture cannot support an identity-accuracy claim. This single release run is separate from the historical timing benchmark.

## Automated checks

~~~powershell
python -m pytest -q tests scripts/evaluation -rs
node --experimental-default-type=module --test tests/*.mjs
python -m compileall -q app.py api face_engine scripts/evaluation tests
git diff --check
~~~

**Baseline Python 95 passed, 0 failed, 2 skipped** in the isolated preferred source snapshot. The two skipped evaluation tests require Windows directory symlinks (WinError 1314). The evaluator-only subset returned **22 passed, 1 deselected** because the omitted test needs SCRFD dependencies. **JavaScript 6 passed, 0 failed.** Module syntax, Python import and compilation, and Git whitespace checks passed. Archify 3.3.0 regenerated the [interactive architecture diagram](../architecture/facefind.html). Validation, delivery, strict check and the diagram's automated browser gate passed.

## Final hardening verification

The following advisory ranges were checked against published upstream advisories on November 24, 2026.

| Package | Previous to hardened pin | Advisory and first fixed version |
|---|---|---|
| Pillow | 10.4.0 to 13.0.0 | TIFF decoder out-of-bounds read and TIFF/EXIF metadata memory exhaustion, both fixed in 13.0.0. [TIFF decoder advisory](https://github.com/advisories/GHSA-c6hw-5grf-ff96), [TIFF/EXIF advisory](https://github.com/advisories/GHSA-wqgw-9xfh-qf2g). |
| Starlette | 0.46.2 to 1.10.0 | Blocking spool rollover, fixed in 0.47.2. Request URL path construction from Host, fixed in 1.0.1. Windows static UNC path resolution, fixed in 1.1.0. [Rollover advisory](https://github.com/advisories/GHSA-2c2j-9gv5-cj73), [Host advisory](https://github.com/advisories/GHSA-86qp-5c8j-p5mr), [Windows static-path advisory](https://github.com/advisories/GHSA-wqp7-x3pw-xc5r). |
| python-multipart | 0.0.20 to 0.0.34 | Preamble/epilogue processing, fixed in 0.0.26. Part header bounds, fixed in 0.0.27. Negative Content-Length in the library's standalone `parse_form` helper, fixed in 0.0.31. FaceFind uses Starlette's streaming parser. [Preamble and epilogue advisory](https://github.com/advisories/GHSA-mj87-hwqh-73pj), [Header advisory](https://github.com/advisories/GHSA-pp6c-gr5w-3c5g), [Standalone parser advisory](https://github.com/advisories/GHSA-v9pg-7xvm-68hf). |
| FastAPI | 0.115.12 to 0.165.0 | Companion upgrade for the Starlette pin. Version 0.165.0 declares Python 3.11+ and Starlette 1.6.0+. [Published dependency metadata](https://raw.githubusercontent.com/fastapi/fastapi/0.165.0/pyproject.toml). |

The base and optional manifests share the patched pins through `requirements.txt`. Fresh Windows x64 Python 3.11.9 package directories resolved and ran the application with NumPy 1.26.4 and OpenCV contrib 4.10.0.84. The optional installation also ran ONNX Runtime 1.32.0, InsightFace 2.1, and scikit-image 0.26.0. InsightFace still declares a different OpenCV distribution name. The documented separate `--no-deps` installation avoids a second `cv2` provider, and the metadata warning remains expected.

Starlette's rollover is offloaded to a worker. A regression test paused rollover on an ordinary 640 x 640 PNG while both health and status responded. Parsing still enforces streamed aggregate bounds and closes an incomplete file even if Starlette has yet to attach it to the parsed form. Decoder tests covered the five baseline formats and AVIF, EXIF orientation, grayscale and RGBA conversion, truncation, invalid content, decoded-pixel limits, mixed batches, per-file caps and multipart cleanup. HEIC/HEIF was unavailable. The tests used ordinary images or invalid and truncated fixtures. No exploit payloads were exercised.

The same-stem replacement and queue advancement regression now rejects stale IDs with HTTP 409. Changing queue position by skipping preserves the embedding revision. Actual enrollment and correction advance it, and unused automatic correction can be kept without advancing it. Successful preview, crop, thumbnail and annotated analysis responses send `Cache-Control: no-store`. Model errors redact checkout and other local paths before reporting to the browser.

Tests confirmed distinct ASCII and accented-Latin JPEG labels, bounded placement, long name handling, Unknown rendering, and stable numbered fallback. Bengali/Hindi shaping tests skipped on the Windows wheel. The JPEG still identifies the face by number and the results and CSV retain its Unicode name. A font with suitable glyphs and a Pillow build with RAQM are needed for shaped labels. This release ships no fonts.

Archify 3.3.0 regenerated the diagram from the updated specification. Validate, deliver, check and browser-check passed. The diagram browser gate checks Archify. Separate Chrome application workflow evidence is reported above.

| Hardened check | Outcome |
|---|---|
| Full optional Python environment | 103 passed, 0 failed, 4 skipped. Two Windows directory-symlink skips and two missing Indic shaping skips. |
| Clean base Python upload and gallery subset | 30 passed, 0 failed, 2 skipped for Indic shaping, 1 research-only test deselected. |
| Desktop JavaScript | 6 passed, 0 failed. |
| Bundled-only HTTP | Requested SCRFD + AdaFace and automatically started YuNet + SFace. Enrolled six identities after one manual confirmation. Group analysis returned an annotated JPEG with one assigned identity and 12 Unknown. A failed switch preserved the current gallery. |
| Preferred HTTP | Started SCRFD + AdaFace. Enrolled 52 identities with nine manual confirms. There were two automatic selections. Strict review and automatic correction passed. Two concurrent requests for the same pending ID produced one enrollment and one HTTP 409. Both pipelines restored their own cache after switching. Repeated group analysis reused embeddings and returned eight assigned identities and six Unknown. These counts are model outputs on an unannotated photograph. |
| Privacy and offline boundary | Live API checked no-store on thumbnails, interactive and automatic review previews, group previews and annotated responses. Python DNS and TCP were guarded in the isolated server processes, with no external attempts during replay. This does not cover native-code networking or replace a disconnected browser demonstration. |
| Rendering environment | Two Indic shaping tests skipped on the recorded build. See current operational constraints above. |
