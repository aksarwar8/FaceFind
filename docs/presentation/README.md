# FaceFind technical presentation

This optional interview companion presents Industry.AI Task A, *Who Is in the Photo?* It describes the published application release `9734b203f524670b695e85b0ea2088f4c6794517`.

- [Presentation PDF](FaceFind_Technical_Presentation.pdf)
- [Editable PowerPoint](FaceFind_Technical_Presentation.pptx)

## Interview timing

Use slides 1–8 for a three-minute introduction, then allow about seven minutes for the live workflow. Keep the remaining time in a 20-minute interview for questions. Slides 9–11 cover research cohorts, evaluation methodology, and reproducibility.

## Evidence

- [Research report](../../RESEARCH.md), [sanitized aggregates](../research/benchmark-aggregates.json), and [research figures](../research/figures/) support the model comparisons and synthetic demonstration. The slide chart uses F1 values from the same 500-image detector cohort. The application latency comes from a separate five-warm-request comparison on one genuine group photo.
- [System architecture](../../ARCHITECTURE.md) and [genuine Archify output](../architecture/facefind.html) support the editable architecture diagram.
- [Approved screenshot register](../screenshots/README.md), [release verification](../verification/RELEASE.md), and [collaboration evidence](../../AI_COLLAB.md) support the workflow, test counts, and ownership discussion.

The 56/57 result applies to five synthetic composites with approximate boxes. Four demo gallery identities overlap with the calibration gallery. The separately documented offline acceptance test was performed by the developer. The screenshots illustrate application states.

## Regenerate the files

From the repository root, run the following in Windows PowerShell on a system with desktop Microsoft PowerPoint and Arial installed.

~~~powershell
.\docs\presentation\build-presentation.ps1
~~~

The script authors native editable PowerPoint text, shapes, bars and table cells and embeds approved PNG screenshots from the repository. It checks the text bounds and exports the PDF from the same deck. To render separate PNG slides, pass `-RenderDirectory` with a local output path. No model files, runtime dependencies or third-party font binaries are added.
