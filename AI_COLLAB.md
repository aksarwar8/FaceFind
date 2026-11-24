# Human-directed AI engineering in FaceFind

[Original conversations](#ai-tools-and-original-conversation-records) · [Engineering ownership](#human-engineering-ownership) · [Timeline](#engineering-timeline) · [Nine decisions](#nine-principal-decisions) · [Authentic exchanges](#authentic-conversation-evidence) · [Review](#independent-review-register) · [Archive coverage](#original-ai-conversation-archive-and-coverage)

I defined FaceFind as an offline CPU workflow for enrolling named references, identifying people in group photos, and leaving uncertain or unenrolled faces Unknown. I used AI for research reasoning, architectural exploration, experiment implementation, software development, debugging, independent review, verification and documentation. I set the operating constraints, challenged assumptions with application observations, specified acceptance checks and authorized release. AI contributed substantial implementation and testing. The decisions below show how I used those contributions to choose what to ship.

## AI Tools and Original Conversation Records

Industry.AI Task A permits authentic exports or accessible share links. I opened the ChatGPT and Claude links without signing in, checked their contents for privacy and approved publication. [History index and verification notes](docs/ai_chats/README.md)

| Tool | Engineering role | Evidence | Original conversation |
|---|---|---|---|
| ChatGPT | Research reasoning, interpretation and planning, as credited in the project record | Research guidance is documented at the role level. This document attributes specific benchmark or code actions to the preserved record that establishes them. | [ChatGPT shared conversation](https://chatgpt.com/share/6b05b0fa-83d8-83ee-ac66-abccda7e8bb6) |
| Codex | Source implementation, experiments, debugging, regression tests, verification and documentation | Dated user instructions, AI responses and selected terminal evidence. The methodology, detector and evaluation exchanges below are directly traceable. | [Codex transcripts and index](docs/ai_chats/codex/README.md) |
| Claude | Independent critique of startup, rights, methodology and documentation | Developer-relayed findings and subsequent corrections are distinguished from the tests that verified them. | [Claude shared conversation](https://claude.ai/share/fbf921c0-3767-4289-a890-63b08d9ae486) |

## Human engineering ownership

**My responsibilities.** I framed the open-set problem, set the local-runtime and publication constraints, chose between proposed fixes, and required controlled comparisons when practical observations contradicted benchmark expectations. I reported enrollment and browser interaction failures, specified independent correctness checks, and performed the disconnected-network acceptance check. [Decisions](#nine-principal-decisions) · [Developer acceptance](docs/verification/RELEASE.md#developer-performed-disconnected-network-acceptance)

**AI contributions.** The tools helped formulate experiments and alternatives. Codex implemented source changes, executed controlled runs and regression tests, and inspected release behavior. I used independent critique to challenge the design, then accepted corrections against code, saved measurements or explicit verification. [Review register](#independent-review-register) · [Release evidence](docs/verification/RELEASE.md)

## Collaboration workflow

~~~mermaid
flowchart TD
    Define["Developer defines problem and constraints"] --> Plan["AI-assisted research and test plan"]
    Plan --> Build["Implement or reproduce with AI"]
    Build --> Test["Run controlled checks and record evidence"]
    Test --> Verify["Developer reviews evidence and checks workflow"]
    Verify --> Critique["Independent critique of design and claims"]
    Critique --> Correct["Correct implementation, method or documentation"]
    Correct -->|"Retest when behavior changes"| Test
    Verify -->|"Unexpected behavior"| Plan
    Correct -->|"Evidence meets acceptance criteria"| Decide["Developer decides and authorizes release"]
~~~

The process repeated across decisions. The scale diagnosis began with a manual discrepancy, the open-set correction began with an audit of saved outputs, and Evaluation Studio began with a product question. [Original exchanges](#authentic-conversation-evidence)

## Engineering timeline

Dates come from commits or dated records. Related investigations are grouped when their exact sequence is secondary to the evidence.

| Phase and objective | Human direction | AI work and verified outcome |
|---|---|---|
| Nov 21–23, 2026 · Local baseline | Define offline enrollment, group-photo recognition and Unknown outcomes. | The committed recognition engine grew into a modular browser application. [Engineering chronology](docs/verification/MILESTONES.md) |
| Detector and recognizer research | Compare common inputs and weigh quality against CPU cost. Preserve incomplete experiments as partial. | WIDER FACE and LFW comparison runs documented quality and latency trade-offs. The shared YOLO comparison covered 1,500 images. [Cohort register](RESEARCH.md#experimental-cohort-register) |
| Nov 23 · Repair open-set methodology | Require every known probe to have an eligible gallery identity and keep production thresholds fixed. | Codex audited v1, implemented preflight and checked gallery agreement before scoring. V2 enrolled all 200 known identities. [Methodology correction](RESEARCH.md#correcting-the-open-set-cohort) |
| Nov 23–24 · Integrate and harden two pipelines | Keep the preferred research bundle and require an independently runnable evaluator path. Fix confirmed review findings while preserving recognition behavior. | Model adapters and gallery caches supported selection and fallback. Regression checks covered bounded intake and stale review handling. [Startup design](ARCHITECTURE.md#model-initialization-and-fallback) · [Hardening](docs/verification/RELEASE.md#final-hardening-verification) |
| Nov 24 · Resolve practical enrollment failures | Investigate missed references and a false second candidate. Keep review under the operator’s control. | Scale controls supported a gallery-only retry. The brick-wall diagnosis and smaller-target stress cases established where Smart selection needs oversight. [Engineering case studies](RESEARCH.md#engineering-case-studies) |
| Nov 24 · Verify actual browser behavior and correctness | Require direct selection of any pending card and independent scoring after Analyze. | Chrome verified queue handling and stale recovery. Evaluation Studio froze inference, checked annotation association and exported outcomes. [Browser verification](docs/verification/RELEASE.md#evaluation-studio-verification) |
| Nov 24 · Reproduce and publish | Require a base-only public-clone smoke test, protect private material, and address the final independent review. | The bundled-only clone analyzed a photo. The frozen application and later documentation corrections were published. [Clone verification](docs/verification/RELEASE.md#public-clone-verification) · [Final review corrections](https://github.com/aksarwar8/FaceFind/commit/5cb4fcb9fa6b88e4758453ca8ea026d0eb82fdd4) |

## Nine principal decisions

First-person decisions are paraphrases of documented developer directions. The original wording is retained in the excerpts.

### 1. Keep two independently usable pipeline choices

**AI suggested:** After I directed the app to keep the bundled fallback, Codex traced startup failure to model construction before fallback or status existed. It proposed trying the requested bundle first, reporting fallback and keeping diagnostics reachable if both failed. [Startup audit exchange](https://github.com/aksarwar8/FaceFind/blob/4309def70781bcb2d4eee29ce78b9352fae2be28/docs/ai_chats/codex/session-2026-11-23T21-02-41.md#L1570-L1575)

**I decided:** Keep SCRFD + AdaFace as the optional preferred research configuration and YuNet + SFace as the bundled path.

**Why:** The preferred configuration improved measured outcomes on selected cohorts at higher CPU cost. The evaluator needed a working checkout without optional checkpoints and their accompanying rights constraints.

**Evidence:** A base-only public clone started via fallback and analyzed an image. [Model trade-offs](RESEARCH.md#multi-objective-decision) · [Checkpoint terms](models/face/MODEL_LICENSES.md)

### 2. Repair the open-set cohort before interpreting matching

**AI suggested:** Codex found that only 146 of 200 intended known identities enrolled in v1. It proposed preflighting references with the production detector, replacing unsuitable identities from a seeded reserve and recording actual membership.

**I decided:** Require every known probe to match an enrolled identity, keep unknown identities disjoint, preserve v1, and rerun the corrected protocol with unchanged production thresholds.

**Why:** An absent gallery identity makes a correct known assignment impossible. Freezing production rules isolates a methodology change from a matcher change. The repaired cohort has a different composition, so the reports retain separate denominators.

**Evidence:** V2 enrolled 200 known identities and scored a cohort of 756 known probes. [Original exchange](#open-set-benchmark-audit-and-correction) · [Research analysis](RESEARCH.md#correcting-the-open-set-cohort)

### 3. Retry SCRFD only for empty reference detections

**AI suggested:** Codex held score settings constant, isolated effective face scale, and proposed a 640-pixel retry after an empty primary reference result.

**I decided:** Keep the 1,280-pixel primary pass, detector and recognition thresholds, and group-photo path. Apply the retry only during gallery enrollment.

**Why:** A real folder exposed a weakness the general detector benchmark did not predict. The controlled change recovered the missed faces without changing group analysis or lowering the acceptance floor.

**Evidence:** Coverage rose from **14/17 to 17/17** on that diagnostic folder. Those photos establish recovery for the observed misses. [Original exchange](#detector-scale-challenge-and-controlled-retry) · [Scale evidence](RESEARCH.md#a-scrfd-missed-real-references)

### 4. Keep enrollment intent under human control

**AI suggested:** Codex implemented a dominant-face heuristic and exposed candidate crops, scores and correction controls.

**I decided:** Offer Smart selection for convenience and retain strict review and manual correction. Use **Always Review Multiple Faces** for evaluator folders when selection accuracy takes priority.

**Why:** A brick-wall recess produced a false second candidate. Area and score can help prioritize review, but they cannot infer which person the uploader intends to enroll.

**Evidence:** Required reviews fell from three to one in the original folder. All three automatic choices on the designated smaller-target stress cases were wrong. [False-positive diagnosis](RESEARCH.md#b-brick-wall-false-positive) · [Smart validation](RESEARCH.md#c-smart-auto-selection-and-intent)

### 5. Verify pending review through the actual browser

**AI suggested:** Codex reproduced the static cards, traced the working preview API, and implemented click-through, queue advancement and recovery after stale responses.

**I decided:** Require every pending card to open directly. Keep immutable IDs and a refresh path for stale state.

**Why:** A valid endpoint did not make a non-first candidate reachable in the UI. A browser test was needed to distinguish an interaction defect from an API failure.

**Evidence:** Chrome reproduced 57 enrolled and three pending. Direct review, queue advancement and separate two-session HTTP 409 recovery passed. The reported empty preview on a current ID did not reproduce. [Browser evidence](docs/verification/RELEASE.md#evaluation-studio-verification)

### 6. Correct verified reliability defects without changing recognition

**AI suggested:** Codex checked advisory ranges and implemented bounded upload handling, immutable review identifiers and local Unicode labels with a numbered fallback.

**I decided:** Require fixes for confirmed findings while preserving models, weights, matching rules and calibration.

**Why:** A stale ID or ambiguous annotation can produce a misleading outcome even when the model behaves as designed. Upload limits and cleanup protect the local workflow.

**Evidence:** Tests covered same-stem replacement, concurrent pending actions, temporary-file cleanup, spool responsiveness and annotation fallback. [Hardening verification](docs/verification/RELEASE.md#final-hardening-verification)

### 7. Make correctness an independent evaluation

**AI suggested:** Codex implemented a frozen prediction snapshot, image association, deterministic one-to-one matching, outcome categories and separate conditional and end-to-end metrics.

**I decided:** Add Evaluation Studio as an optional second step with independently supplied ground truth or manual review. Keep ordinary Analyze usable without labels.

**Why:** I needed to know whether predicted names were right and whether Unknown meant a correct rejection. Counts alone leave wrong assignments, missed faces and failed processing unresolved.

**Evidence:** Chrome verified scoring and exports. SCRFD + AdaFace produced **56/57 correct outcomes** across five synthetic composites with approximate boxes and a 60-identity gallery. [Original exchange](#evaluation-studio-and-independent-correctness) · [Demo scope and calibration overlap](RESEARCH.md#evaluation-studio-demonstration)

### 8. Test the evaluator path from a public clone

**AI suggested:** Codex used a clean clone and an isolated base runtime to check fallback, enrollment, recognition and Evaluation Studio.

**I decided:** Require that check with only the bundled models and keep recognition usable without ground truth.

**Why:** My development environment could hide checkpoint, package or undocumented setup requirements that an evaluator would encounter.

**Evidence:** The clone matched the frozen commit, used YuNet + SFace, returned an annotated JPEG and showed **Not Evaluated** in the evaluation workspace. [Public clone verification](docs/verification/RELEASE.md#public-clone-verification)

### 9. Publish reviewable evidence with privacy and rights boundaries

**AI suggested:** Codex separated publication files from private material and generated figures from sanitized aggregates.

**I decided:** Publish the frozen application, research, Archify artifacts and verified documentation. Keep transcripts authentic and publish only the reviewed disclosure set.

**Why:** Evidence must be traceable without publishing private photographs, biometric embeddings, credentials or restricted checkpoints.

**Evidence:** The review package and documentation audit recorded the publication scope. Bundled checkpoint notices remain attached to the model-rights record. [Audit](docs/verification/DOCUMENTATION-CONSISTENCY.md) · [Rights register](models/face/MODEL_LICENSES.md)

## Authentic conversation evidence

These are selected verbatim excerpts. Speaker labels and dates come from the transcript. Surrounding text is omitted.

### Open-set benchmark audit and correction

November 23, 2026. Codex audited the saved run, then I specified the constraints for the correction.

**Codex, audit recommendation**

> The smallest safe benchmark correction is to preflight gallery-reference images with the production detector, then form the known set from identities whose chosen reference yields exactly one usable face. If an identity has no suitable reference, replace it from a seeded reserve pool.

**Developer, follow-up**

> 5. Every known probe must belong to an identity actually represented in the gallery.
> 6. Keep the 100 unknown identities completely disjoint from all known identities.
> 7. Keep the existing production thresholds unchanged.

The follow-up kept gallery construction and threshold behavior separable. [Original transcript](https://github.com/aksarwar8/FaceFind/blob/4309def70781bcb2d4eee29ce78b9352fae2be28/docs/ai_chats/codex/session-2026-11-23T12-34-09.md#L1453-L1483)

### Detector scale challenge and controlled retry

November 24, 2026. I challenged the benchmark assumption after observing misses in a real gallery.

**Developer, diagnostic instruction**

> Do not assume that SCRFD is inherently superior because of the WIDER FACE benchmark.

**Codex, diagnostic response**

> SCRFD missed all three faces at FaceFind’s default 1,280-pixel detection scale. A **640-pixel retry recovered all three at the same 0.50 threshold**.

**Developer, implementation follow-up**

> 3. Keep the existing detection threshold, NMS threshold and recognition thresholds unchanged.

The change was limited to reference enrollment and supported by a controlled scale comparison. [Original transcript](https://github.com/aksarwar8/FaceFind/blob/4309def70781bcb2d4eee29ce78b9352fae2be28/docs/ai_chats/codex/session-2026-11-24T11-08-36.md#L2283-L2420)

### Evaluation Studio and independent correctness

November 24, 2026. I asked for evidence of whether assignments and rejections were correct. The retained request states the motivation as follows.

**Developer, product requirement**

> Currently, FaceFind reports detected, recognized and unknown faces. These are inference outcomes, not verified correctness metrics.

**Codex, implementation report**

> The studio scores a frozen copy of the completed predictions, and evaluation resets when the analysis, gallery revision, or pipeline changes.

The resulting evaluation stays outside the recognition path. [Developer request](https://github.com/aksarwar8/FaceFind/blob/4309def70781bcb2d4eee29ce78b9352fae2be28/docs/ai_chats/codex/session-2026-11-24T11-08-36.md#L4407-L4450) · [Codex report](https://github.com/aksarwar8/FaceFind/blob/4309def70781bcb2d4eee29ce78b9352fae2be28/docs/ai_chats/codex/session-2026-11-24T11-08-36.md#L4972-L5002)

## Independent review register

The initial independent critique raised startup, evidence and rights questions. The developer relayed those findings and requested targeted fixes. Source inspection, saved runs, regressions and the public-clone smoke test establish the resolutions below. The later independent review examined the documentation snapshot.

| Finding | Disposition and evidence |
|---|---|
| Missing optional weights could break startup | **Corrected.** Fallback and diagnostics were verified. [Public clone](docs/verification/RELEASE.md#public-clone-verification) |
| Upload dependency advisories and blocking spool rollover | **Corrected for recorded versions.** Updated pins and bounded parsing were checked with a responsiveness regression. [Hardening](docs/verification/RELEASE.md#final-hardening-verification) |
| Stale queue index could address a different reference | **Corrected.** Immutable IDs, HTTP 409, refresh and separate embedding revisions were tested. [Lifecycle tests](tests/test_gallery_api_lifecycle.py) |
| Unicode labels could appear identical | **Fallback verified.** Structured results retain the identity. Indic shaping depends on the local build. [Annotation tests](tests/test_annotations.py) |
| Recognition denominators can hide failures | **Method corrected.** V2 reconciles gallery membership. Conditional, end-to-end and synthetic-cohort results retain explicit scope. [Research](RESEARCH.md#correcting-the-open-set-cohort) |
| Pipeline attribution, collaboration evidence and demo guidance | **Documentation corrected.** The later review used `5a524db`. Publication clarified the SCRFD + AdaFace result, separated research and deployed fallback rules, and added strict-review guidance and a verbatim exchange. [Correction commit](https://github.com/aksarwar8/FaceFind/commit/5cb4fcb9fa6b88e4758453ca8ea026d0eb82fdd4) |
| Checkpoint rights and evaluation transfer | **Qualified.** Bundled notices are retained. Optional checkpoint terms and provenance remain separate. Generalization to unseen genuine group photos still needs validation. [Rights register](models/face/MODEL_LICENSES.md) · [Future validation](RESEARCH.md#future-research-and-validation) |

## Original AI Conversation Archive and Coverage

The archive includes one developer-verified share for each of [ChatGPT](https://chatgpt.com/share/6b05b0fa-83d8-83ee-ac66-abccda7e8bb6) and [Claude](https://claude.ai/share/fbf921c0-3767-4289-a890-63b08d9ae486), plus [seven recovered Codex transcripts](docs/ai_chats/codex/README.md). These are the original-record formats permitted by the assignment.

The Codex transcripts contain **448 messages** across **103 user prompts and 345 Codex messages**, plus 59 terminal excerpts. The latest included message is **2026-11-24 17:31:29.732 UTC**. The Markdown comes from local JSONL records and retains speaker, time and order within each session. It marks privacy redactions, normalizes trailing whitespace and excludes private images, sensitive material and internal records. The original logs remain private.

**Coverage is partial.** The first Codex development prompt remains unrecovered. The share links establish access to their respective conversations. Earlier and separate ChatGPT or Claude threads, and later turns beyond the archived Codex cutoff, do not have established coverage here. [Submission evidence](docs/verification/MISSING-EVIDENCE.md#required-submission-evidence)

## Engineering reflection

I learned to ask what each measurement actually establishes. The v1 gallery mismatch changed what a known probe could measure, while the reference misses showed why a detector comparison could not settle application behavior. I kept calibration fixed while correcting the cohort, then required a separate check for the retry. I kept manual oversight because detector confidence and face area cannot express the uploader’s intent.

I treated AI-generated changes as proposals to check against code, experiments and the UI. The pending-card defect needed a browser reproduction, and Evaluation Studio needed independent truth before its counts could mean correctness. I applied the same standard to release: a clean clone checked reproducibility, a disconnected test checked local operation, and the rights and privacy review determined what could be published.
