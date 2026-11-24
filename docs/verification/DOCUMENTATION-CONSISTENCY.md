# Documentation consistency report

**Frozen application** [`9734b203f524670b695e85b0ea2088f4c6794517`](https://github.com/aksarwar8/FaceFind/commit/9734b203f524670b695e85b0ea2088f4c6794517)

**Documentation audit baseline** [`5a524dba68760c3ba7502e077d9af09ba34b25d7`](https://github.com/aksarwar8/FaceFind/commit/5a524dba68760c3ba7502e077d9af09ba34b25d7), tree `aeacc950e06a35633abe5de2c8a8ce5d25743782`.

This audit reconciles the public submission with the committed application, preserved experimental outputs, browser and offline evidence, and original visible collaboration records.

| Audit area | Finding and disposition |
|---|---|
| Release provenance | Earlier installer and browser milestones were being read as current. The product guide, system design and verification register now identify the published Evaluation Studio application and distinguish its public-clone smoke test from earlier snapshots. |
| Detector comparability | The standalone YuNet and three-detector runs share all 500 image IDs. The common-input run adds bottom/right padding; it reports 3,338 TP and 987 FP versus the standalone run's 3,342 and 983. Both rows are retained. |
| Recognition denominators | The early known-gallery mismatch and reported lower/upper Possible settings are described separately. The recognition-only legacy rejection complement includes unprocessed probes. True rejection rates now use actual evaluated outcomes and stated attempted or evaluated denominators. |
| Calibration and independent truth | Frozen values and production constants were cross-checked. The 60-way final demo uses parameters from 50-way calibration and shares some identities with both earlier calibration pools. The report publishes counts only and keeps exact image and pretraining overlap unresolved. |
| App timing | Model and detector timing are grouped with the appropriate cohorts and units. Application stage means were summed and both unexplained remainders reported as differences. |
| Demonstration results | Saved Chrome reports and annotations confirm the 60-identity gallery, five composites and 56/57 verified outcomes. No accuracy claim was made for the unannotated genuine-photo timing fixture. |
| Architecture and resilience | Limits, revisions, cache behavior, API and evaluation semantics were checked against implementation. The published specification and Archify HTML are preserved byte for byte. The editable specification already includes the independent evaluation path. |
| Human and AI attribution | The published ChatGPT and Claude share links were checked manually by the developer. Seven sanitized Codex transcripts include the completed presentation turn through 17:31:29.732 UTC on November 24, 2026. Exact scope and evidence gaps are indexed. |
| Privacy and rights | Six approved redacted application screenshots and previously checked presentation assets are included. Raw photos, recordings, private speaker notes, environments and optional checkpoints remain outside tracked deliverables. Only YuNet and SFace checkpoint binaries are tracked. |
| Browser and offline evidence | Current Chrome workflow evidence supersedes historical browser-access and download gaps. Private OBS footage establishes developer-performed native folder selection. The disconnected-network acceptance is developer-performed and its still shows startup. |
| Presentation | The published 11-slide PPTX, PDF and generation source retain the verified synthetic, detector and timing cohorts, model-rights qualifications, and current test counts. Presentation binaries and Archify artifacts are preserved unchanged. |
| Submission evidence | The recovered requirements summary maps to the compliance matrix. Full original AI history remains mandatory and incomplete. [Outstanding evidence](MISSING-EVIDENCE.md#required-submission-evidence) |

The final recorded suites returned **103 Python passes, four documented skips and 14 JavaScript passes**. The skips were two Windows directory-symlink and two Indic shaping checks. Those runs and the original benchmark data establish the released evidence. This audit checked static links, archive scope, numerical consistency, source provenance and unchanged application files. No model benchmarks, automated suites or browser workflows were rerun.

[Evidence index](../research/EVIDENCE.md) · [Publication verification](RELEASE.md) · [Compliance matrix](COMPLIANCE.md)
