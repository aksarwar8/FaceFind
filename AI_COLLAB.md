# Human-directed AI engineering in FaceFind

[Roles](#roles-and-evidence) · [Workflow](#collaboration-workflow) · [Timeline](#engineering-timeline) · [Nine decisions](#nine-principal-decisions) · [Review register](#independent-review-register) · [Original excerpts](#authentic-conversation-evidence) · [Remaining evidence](#full-history-evidence-still-needed)

## Roles and evidence

The developer directed the product and its operating constraints, challenged model choices with local observations, defined the browser experience, requested review and authorized publication. AI assistance was substantial. It implemented and tested the software, built experiments, investigated failures and helped turn measured evidence into decisions.

| Participant | Role and evidence |
|---|---|
| Human developer | Set the offline CPU goal, required genuine evidence and model-rights boundaries, decided which pipeline to retain, surfaced real enrollment and browser failures, and required independent correctness metrics. The project conversation records these directions. |
| ChatGPT | Research guidance, technical reasoning, experiment interpretation and planning are credited in the project record. The developer checked the published share link. [Conversation record](docs/ai_chats/chatgpt/README.md) |
| Codex | Inspected source and artifacts, implemented the application and tests, executed controlled comparisons and diagnostics, verified integration and browser behavior, built genuine Archify output, and performed release and clone checks. Source, saved results and project exchanges support this work. |
| Claude | Independent critique of startup, methodology, model rights and documentation informed the review register. The developer checked the published share link. [Conversation record](docs/ai_chats/claude/README.md) |

## Collaboration workflow

~~~mermaid
flowchart LR
    H[Human observes a limitation] --> Q[Define a testable question with AI]
    Q --> T[Implement or run a controlled test]
    T --> E[Record measured evidence and failures]
    E --> R[Review scope, denominators and reproducibility]
    R --> D[Human directs the next decision]
    D --> H
~~~

The detector investigation shows this loop. The benchmark favored SCRFD, but a real gallery exposed misses. Codex isolated scale from thresholding and NMS. The developer directed a gallery-only retry at the demonstrated scale while preserving model behavior elsewhere. The changed application then needed its own verification. [Scale investigation](RESEARCH.md#a-scrfd-missed-real-references)

## Engineering timeline

This chronology is reconstructed from the checked Git history, run identifiers and the available project exchanges. The order is thematic where individual experiment timestamps or original conversations are unavailable.

| Phase | Human direction or review | Engineering evidence |
|---|---|---|
| Requirements and baseline | Specify a local face-gallery and group-photo workflow on CPU. | Initial application commits on November 21–23, 2026, with the modular YuNet + SFace baseline. |
| Detector investigation | Compare on common inputs and retain partial runs truthfully. | WIDER FACE checked comparisons and the preserved shared YOLO prefix. |
| Recognition and calibration | Evaluate alternatives for accurate known recognition and unknown rejection under CPU constraints. | LFW comparisons, per-model calibration and recognition cost. |
| Methodological repair | Require consistent ground truth and explicit denominators. | V2 preflights references before considering an identity known. The v2 preflight and consistency checks are evidenced by the [research record](RESEARCH.md#correcting-the-open-set-cohort). |
| Preferred pipeline and desktop design | Keep SCRFD + AdaFace as the preferred research bundle and require a usable offline browser application. | Model adapters, fallback, versioned galleries and desktop workflows. |
| Independent review and hardening | Request external critique, then require reproducible startup, rights clarity and defensive local behavior. | Review concerns recorded in the owner brief; verified code, dependency and regression changes. |
| Real-world diagnostics | Challenge missed reference faces and an implausible second candidate. | Controlled scale experiment and a brick-wall false-positive investigation. |
| Human-in-the-loop enrollment | Reduce review effort while preserving user intent and correction. | Smart and strict policies, smaller-target stress failures and immutable review IDs. |
| Actual browser testing | Report that visible pending cards could not be selected directly. | Browser reproduced 57 enrolled and three pending. Pending cards were static. The patch added click-through, automatic advancement and recovery after a stale preview. |
| Evaluation Studio | Ask how to prove that a recognized or unknown face was correctly classified. | Independent annotation, deterministic matching, outcome matrix and report exports. |
| Release and documentation | Authorize publication, then require fresh-clone verification and a documentation-only refinement. | Public application baseline and targeted Chrome release evidence. [Release verification](docs/verification/RELEASE.md) |

## Nine principal decisions

The first-person fields below summarize documented owner directions. The AI fields describe verified proposals or implementation responses. Direct quotations appear in the excerpt groups.

### 1. Preserve a runnable bundled pipeline

**AI suggested:**

After I directed the app to keep the bundled fallback, Codex traced the startup failure to model initialization before fallback or status existed. Codex proposed attempting the requested pipeline first, reporting fallback, and keeping health and status reachable if both bundles failed. [Startup audit exchange](https://github.com/aksarwar8/FaceFind/blob/4309def70781bcb2d4eee29ce78b9352fae2be28/docs/ai_chats/codex/session-2026-11-23T21-02-41.md#L1570-L1575)

**I decided:**

Keep SCRFD + AdaFace preferred for local research use subject to the checkpoint terms and retain YuNet + SFace as an independently runnable bundled option.

**Why:**

The research bundle improved measured outcomes on particular cohorts. The evaluator also needed a usable checkout without restricted weights or optional dependencies.

**Evidence:**

A clean public clone started the bundled pipeline and analyzed a photo. [Release verification](docs/verification/RELEASE.md) · [Model decision](RESEARCH.md#multi-objective-decision)

### 2. Score known probes only against a valid gallery

**AI suggested:**

Codex audited the saved v1 run and proposed preflighting references with the production detector, requiring one usable face per reference, and replacing unsuitable identities from a seeded reserve. It also proposed recording actual gallery membership. [Original audit and follow-up](https://github.com/aksarwar8/FaceFind/blob/4309def70781bcb2d4eee29ce78b9352fae2be28/docs/ai_chats/codex/session-2026-11-23T12-34-09.md#L1453-L1483)

**I decided:**

Keep the original and corrected experiments separate and use the corrected cohort for interpretable known-person results.

**Why:**

In v1, only 146 of 200 selected identities enrolled. A known probe without its corresponding gallery identity could never succeed, and the repaired cohort changed composition.

**Evidence:**

The v2 report had 200 enrolled identities and 756 known probes. [Methodological correction](RESEARCH.md#correcting-the-open-set-cohort)

### 3. Investigate the three SCRFD misses at controlled scales

**AI suggested:**

Codex proposed a 640-pixel gallery-only SCRFD pass after the standard pass found zero faces.

**I decided:**

Keep the detector and existing score threshold. Use the smaller pass only for empty gallery detections.

**Why:**

A real reference folder exposed misses despite the detector's benchmark advantage. The scale control recovered those faces while leaving the primary path intact.

**Evidence:**

Coverage rose from 14/17 to 17/17. [Scale investigation](RESEARCH.md#a-scrfd-missed-real-references)

### 4. Keep human control over ambiguous enrollment

**AI suggested:**

Codex implemented a dominant-face heuristic alongside strict review, exposed candidate crops and scores, and retained correction for automatic selections.

**I decided:**

Offer Smart selection for a lower-effort demo and keep strict review and correction available.

**Why:**

A false second candidate on a brick recess showed that detector output alone could not establish whom the uploader intended to enroll.

**Evidence:**

Smart reduced original reviews from three to one, but all three automatic choices in the designated smaller-target stress cases were wrong. [False-positive investigation](RESEARCH.md#b-brick-wall-false-positive) · [Smart validation](RESEARCH.md#c-smart-auto-selection-and-intent)

### 5. Reproduce pending-card behavior in the actual browser

**AI suggested:**

Codex traced working immutable-ID requests and identified the static pending cards. It implemented click-through, queue advancement and stale-preview recovery.

**I decided:**

Require every pending card to open directly and keep immutable IDs and a refresh path after stale requests.

**Why:**

A valid preview endpoint still left the workflow incomplete when the person testing the UI could not open a non-first item.

**Evidence:**

Chrome reproduced 57 enrolled and three pending. Direct review, queue advancement and a separate two-session HTTP 409 recovery passed. The reported empty preview on a valid ID was not reproduced. [Browser verification](docs/verification/RELEASE.md#evaluation-studio-verification)

### 6. Apply reviewed hardening without changing recognition rules

**AI suggested:**

Codex checked the upload-advisory version ranges and implemented bounded upload handling, immutable reference IDs and local Unicode labels with a numbered fallback.

**I decided:**

Require the security and correctness fixes while preserving model weights, matching behavior and calibration.

**Why:**

A stale selection or ambiguous exported label can undermine a local demonstration independently of model quality. Upload handling also needs explicit limits and cleanup.

**Evidence:**

The release checks cover same-stem replacement, concurrent pending actions, temporary-file cleanup, spool responsiveness and annotation fallback. [Hardening verification](docs/verification/RELEASE.md#final-hardening-verification)

### 7. Evaluate correctness with independent ground truth

**AI suggested:**

Codex separated scoring from inference and implemented deterministic one-to-one box matching, an outcome matrix and separate conditional and end-to-end metrics.

**I decided:**

Add Evaluation Studio as an optional step after recognition, using manual labels or CSV/JSON annotations and frozen predictions.

**Why:**

Recognized and Unknown counts describe predictions. They cannot establish whether an assignment is correct, a known face was rejected, or a face was missed.

**Evidence:**

Chrome verified the independent evaluation flow. Five synthetic composites with approximate annotations yielded 56/57 correct outcomes, including one known rejection. [Measured outcomes](RESEARCH.md#evaluation-studio-demonstration) · [Evaluation design](ARCHITECTURE.md#evaluation-studio)

### 8. Verify the evaluator experience from a public clone

**AI suggested:**

Codex used a clean clone of the published commit and an isolated base runtime to check startup, enrollment, inference and access to Evaluation Studio.

**I decided:**

Require a genuine public-clone smoke test with only bundled models, and leave the ordinary Analyze workflow usable without ground truth.

**Why:**

Development-only packages, weights and files could hide problems that an evaluator would encounter.

**Evidence:**

The clone matched the application commit, used YuNet + SFace, returned an annotated JPEG and opened the evaluation workspace as Not Evaluated. [Public clone verification](docs/verification/RELEASE.md#public-clone-verification)

### 9. Make publication traceable without sharing private evidence

**AI suggested:**

Codex separated publication files from private artifacts and generated quantitative figures from sanitized aggregate inputs.

**I decided:**

Publish the frozen application and its evidence-linked documentation. Preserve original chat meaning and disclose gaps in the submission history.

**Why:**

The evaluator needs reproducible evidence, while credentials, biometric inputs, unreviewed transcripts and restricted checkpoints need to stay private.

**Evidence:**

The documentation audit, source-only review package and chat-history index record the verification scope and remaining gaps. [Consistency report](docs/verification/DOCUMENTATION-CONSISTENCY.md) · [History index](docs/ai_chats/README.md)

## Independent review register

| Finding or concern | Source and verification | Resolution and evidence | Status |
|---|---|---|---|
| Missing optional weights threaten the startup path | Owner-reported independent review; fresh-clone behavior checked directly. | Automatic fallback preserves both choices. The public clone ran without optional checkpoints. [Release verification](docs/verification/RELEASE.md) | Verified resolution |
| Checkpoint terms and training provenance need separate treatment | Owner-reported review concern checked against the retained rights register and published distribution. | Bundled notices retained and restricted or unresolved weights excluded. [Rights register](models/face/MODEL_LICENSES.md) | Rights uncertainty remains |
| Vulnerable upload dependencies and blocking spool rollover | Owner hardening brief; advisories and known version ranges were checked during hardening. | Updated pins, bounded parsing and a responsiveness regression. [Advisory evidence](docs/verification/RELEASE.md#final-hardening-verification) | Verified for recorded versions |
| Queue position can identify a different photo after replacement | Owner relayed a reproduced review finding; regression exercised same-stem replacement and concurrent requests. | Immutable review IDs, stale 409, queue refresh and separate embedding revision. [Lifecycle tests](tests/test_gallery_api_lifecycle.py) | Verified resolution |
| Unicode labels could look identical | Owner relayed a reproduced review finding; the current tests exercised local rendering and numbered fallback. | Local Pillow rendering with structured Unicode identity retained. Indic shaping needs a compatible build. [Annotation tests](tests/test_annotations.py) | Verified fallback |
| Recognition denominators can hide failures | Saved experiments and original code establish the v1 gallery mismatch and the legacy rejection-field interpretation. | V2 gallery consistency, explicit reporting and independent Studio scoring. [Methodological correction](RESEARCH.md#correcting-the-open-set-cohort) | Verified method changes |
| Technical depth and evidence separation were hard to follow | Owner relayed the documentation review's scope and recommendations. The complete critique was unavailable. | Separate product, research, architecture and collaboration narratives, with source-linked figures and a verification record. [Consistency report](docs/verification/DOCUMENTATION-CONSISTENCY.md) | Documentation review pending |

The submitted Gallery capture shows the final state after enrollment, with 60 enrolled and zero pending. The interactive review evidence comes from a separate Chrome run that opened a non-first card, checked the three-item queue and confirmed recoverable handling of a stale preview. [Gallery capture](docs/screenshots/reference-gallery.png) · [Browser verification](docs/verification/RELEASE.md#evaluation-studio-verification)

## Authentic conversation evidence

These are three verbatim excerpts from the available project conversation, with speaker and context identified. They are selected statements with surrounding context omitted. They are not a reconstructed transcript or an external assistant attribution.

**Developer — Evaluation Studio request**

> Currently, FaceFind reports detected, recognized and unknown faces. These are inference outcomes, not verified correctness metrics.

**Open-set benchmark correction · November 23, 2026**

**Codex, from the audit recommendation**

> The smallest safe benchmark correction is to preflight gallery-reference images with the production detector, then form the known set from identities whose chosen reference yields exactly one usable face. If an identity has no suitable reference, replace it from a seeded reserve pool.

**Developer, from the follow-up implementation instructions**

> 5. Every known probe must belong to an identity actually represented in the gallery.
> 6. Keep the 100 unknown identities completely disjoint from all known identities.
> 7. Keep the existing production thresholds unchanged.

[Original exchange](https://github.com/aksarwar8/FaceFind/blob/4309def70781bcb2d4eee29ce78b9352fae2be28/docs/ai_chats/codex/session-2026-11-23T12-34-09.md#L1453-L1483)

**Codex verification report**

> The results apply to five synthetic composites with approximate boxes. They do not establish accuracy on genuine group photographs.

The retained original exchanges document the requested outcomes, the browser-reported failure, the controlled evaluation and the publication authorization. The developer manually opened both shared histories without signing in, reviewed their contents for privacy, and approved publication. Earlier automated requests did not expose the conversation text. [ChatGPT conversation](https://chatgpt.com/share/6b05b0fa-83d8-83ee-ac66-abccda7e8bb6) · [Claude conversation](https://claude.ai/share/fbf921c0-3767-4289-a890-63b08d9ae486)

## Full-history evidence still needed

The recovered assignment brief requires the full project-related AI history from the first prompt. The three selected excerpts, seven recovered Codex sessions, and two shared conversations document available evidence. The first Codex development prompt remains unrecovered, and coverage of any earlier or separate ChatGPT and Claude threads has not been established. [History index](docs/ai_chats/README.md)

## Reflection

AI assistance accelerated implementation, experimental tooling and regression coverage. It also made verification essential. The initial known-probe denominator exposed how a plausible benchmark could answer the wrong question. Independent review brought attention to startup and engineering risks. The developer's manual observations exposed scale sensitivity, a false candidate and an interaction problem that the tests had missed. Evaluation Studio follows the same principle. A prediction becomes a verified outcome only when compared to ground truth with a clear protocol and denominator.
