# Documentation consistency report

**Frozen application** [`9734b203f524670b695e85b0ea2088f4c6794517`](https://github.com/aksarwar8/FaceFind/commit/9734b203f524670b695e85b0ea2088f4c6794517)

The rewrite follows the owner's supplied summary of Claude's structural recommendations. The complete external review text was unavailable. Factual checks used source, preserved local run outputs, release evidence, and original project messages where available.

| Audit area | Finding and disposition |
|---|---|
| Release provenance | Earlier installer and browser milestones were being read as current. The product guide, system design and verification register now identify the published Evaluation Studio application and distinguish its public-clone smoke test from earlier snapshots. |
| Detector comparability | The standalone YuNet and three-detector runs share all 500 image IDs. The common-input run adds bottom/right padding; it reports 3,338 TP and 987 FP versus the standalone run's 3,342 and 983. Both rows are retained. |
| Recognition denominators | The early known-gallery mismatch and reported lower/upper Possible settings are described separately. The recognition-only legacy rejection complement includes unprocessed probes. True rejection rates now use actual evaluated outcomes and stated attempted or evaluated denominators. |
| Calibration and independent truth | Frozen values and production constants were cross-checked. The 60-way final demo uses parameters from 50-way calibration and shares some identities with both earlier calibration pools. The report publishes counts only and keeps exact image and pretraining overlap unresolved. |
| App timing | Model and detector timing are grouped with the appropriate cohorts and units. Application stage means were summed and both unexplained remainders reported as differences. |
| Demonstration results | Saved Chrome reports and annotations confirm the 60-identity gallery, five composites and 56/57 verified outcomes. No accuracy claim was made for the unannotated genuine-photo timing fixture. |
| Architecture and resilience | Limits, revisions, cache behavior, API and evaluation semantics were checked against implementation. The published specification and Archify HTML are preserved byte for byte. The editable specification already includes the independent evaluation path. |
| Human and AI attribution | Original project messages support only selected statements. External review and early ChatGPT responsibilities remain labeled as owner-reported. No transcript, quote or first-person owner reflection was fabricated. |
| Privacy and rights | Public outputs contain aggregate figures only. The bundled and optional model rights remain distinct. The review archive omits all checkpoints, private samples, identity manifests, embeddings, screenshots, credentials, environments and unreviewed transcripts. |
| Submission evidence | The recovered brief has been mapped. Original full ChatGPT, Codex and Claude histories remain mandatory. Publishable captures are optional. [Missing evidence](MISSING-EVIDENCE.md) |

The complete existing regression suites returned **103 Python passes, four documented skips and 14 JavaScript passes**, without changes to test sources. The skips were two Windows directory-symlink checks and two Indic shaping checks. Figure generation used an isolated documentation environment. Six generated SVGs were inspected visually, and the internal link and archive checks passed. No benchmark or Chrome workflow was rerun for this rewrite. The application source, model binaries, thresholds, dependencies and published commit were unchanged.

[Evidence index](../research/EVIDENCE.md) · [Publication verification](RELEASE.md) · [Compliance matrix](COMPLIANCE.md)
