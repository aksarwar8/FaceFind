# Industry.AI Task A compliance

The recovered Task A brief, as supplied by the developer, is the basis for this matrix. The repository deadline stated in the brief is **24 November 2026, 11:59 PM IST**. The [frozen application release](https://github.com/aksarwar8/FaceFind/commit/9734b203f524670b695e85b0ea2088f4c6794517) is already public.

## Mandatory requirements

| Requirement | Status | Evidence or outstanding action |
|---|---|---|
| Offline face identification on a local laptop | Developer acceptance verified | The developer disconnected the laptop from the internet and confirmed startup and local operation. The Dashboard capture shows the local backend connected and the preferred engine ready. [Release verification](RELEASE.md#developer-performed-disconnected-network-acceptance) |
| UI folder selection for roughly 50 named references | Implemented and tested | A private developer OBS recording shows native folder selection and enrollment of 24 references. Separate HTTP and Chrome checks exercised 52 and 60 references. [Verification evidence](RELEASE.md#native-reference-folder-enrollment) |
| Group photograph upload | Verified | Chrome and local API workflows exercised upload and analysis. [Release verification](RELEASE.md) |
| Annotated identities and Unknown labels | Verified | Annotated JPEGs and structured known/Unknown outcomes were checked. [Release verification](RELEASE.md) · [Synthetic demo](../../RESEARCH.md#evaluation-studio-demonstration) |
| Results list | Verified | Analyze renders per-face outcomes. Targeted Chrome checks exercised group analysis and result navigation. [Analyze renderer](../../web/analysis.js) · [Release verification](RELEASE.md) |
| README installation and execution instructions | Prepared and verified | Independently runnable bundled path and optional research setup. [Quick start](../../README.md#installation-and-first-run) |
| AI_COLLAB with 5–10 structured decisions | Prepared | Nine principal decisions, each with AI suggested, I decided, Why and Evidence. [Decision record](../../AI_COLLAB.md#nine-principal-decisions) |
| Two to three authentic pasted chat excerpts | Prepared | Three verbatim excerpt groups identify the speaker and context. [Excerpts](../../AI_COLLAB.md#authentic-conversation-evidence) |
| Full original AI chat history starting from the first prompt | Partially documented | Seven Codex transcripts and developer-verified ChatGPT and Claude share links are available. The developer checked unauthenticated access and privacy manually. Automated retrieval did not expose conversation text. The first Codex development prompt remains unrecovered, and full cross-service coverage is unverified. [History index](../ai_chats/README.md) |
| RESEARCH with investigated alternatives, selections, rejections and links | Prepared and checked | Source-linked model and dataset comparisons, operating points, limitations and research figures. [Research report](../../RESEARCH.md) |
| ARCHITECTURE with genuine Archify and short trade-offs | Prepared and checked | Validated standalone Archify artifact, editable specification and explicit table. [Archify](../architecture/facefind.html) · [Architecture trade-offs](../../ARCHITECTURE.md#architecture-trade-offs) |

## Optional enhancements and evidence

Evaluation Studio, the 60-identity synthetic run, interactive ambiguity review, six redacted screenshots, and the 11-slide technical deck supplement the mandatory deliverables. [Screenshot register](../screenshots/README.md) · [Technical presentation](../presentation/README.md)

**Compliance status** Partial. The full original AI history is mandatory and remains incomplete. [Outstanding evidence](MISSING-EVIDENCE.md#required-submission-evidence)
