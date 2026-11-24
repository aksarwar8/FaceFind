# Manual browser evidence

Six approved, privacy-redacted screenshots are published. The evidence inventory below also records the private manual-evaluation captures. They show the following application states. Output images here are re-encoded crops or opaque-masked copies of real captures. Unredacted originals, underlying photographs and identity data remain private.

| Supplied capture | What it shows | Documentation treatment |
|---|---|---|
| 01 Dashboard | Ready engine, 60 references, zero pending, latest analysis and a manual evaluation | [Dashboard](dashboard.png). Photos, names, filename and the manual evaluation are masked. |
| 02 Reference Gallery | Gallery Manager after enrollment, with 60 accepted, zero rejected and zero pending | [Reference Gallery](reference-gallery.png). Portrait grid and identity labels are masked. |
| 03 Recognition Results | Analyze with ten detections and ten predicted identities | [Group-photo recognition](group-photo-recognition.png). The photograph and identity-linked results are masked. These are inference counts. |
| 04 Evaluation Studio | Original composite and evaluated outcome rows populated through manual correspondence | Private. The independently supplied annotations are not verified for this capture. |
| 05 Evaluation Metrics | Manual-correspondence summary for the owner's all-known example | Private. No matching annotation export validates the manual mapping. |
| 06 System Settings | SCRFD + AdaFace shown as preferred, requested and active, with CPU model readiness | [System Settings](system-settings.png). Cropped to the engine controls; the background includes the unverified 5/9 manual evaluation and stays private. |
| 07 Annotated Output | Exported synthetic group composite with boxes, assigned names and Unknown labels | Private. Contains source photo pixels and identities. |
| 08 Offline Demo | Dashboard during the developer-performed disconnected-network test. Local backend connected, SCRFD-10G-KPS + AdaFace IR-18 ready, gallery empty, and no current analysis. | [Offline demo](offline-demo.png). Cropped to the application to remove browser tabs and desktop details. |

## Verified Evaluation Studio capture

The [Evaluation Studio image](evaluation-studio.png) comes from the earlier Chrome run of the Easy synthetic composite. It shows the imported manifest, approximate annotations, one-to-one matching at IoU 0.50 and 9/10 correct outcomes. The stored report is complete and its image hash matches the archived composite. The verified five-group total is 56/57 correct outcomes. The owner's manual-correspondence metrics, including the earlier 55.6% (5/9), are excluded. [Protocol and cohort results](../../RESEARCH.md#evaluation-studio-demonstration)

## Redistribution

Per-image redistribution permission was not established for the LFW-derived photos. The dataset authors' responses report that copyright belongs to the originating publications. Public documentation copies remove photo pixels and gallery names, and remove browser and desktop chrome where present. [LFW dataset authors' responses, Appendix B](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/01/1803.09010.pdf)
