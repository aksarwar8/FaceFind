import { csvCell } from "./exports.js";

export const LIMITATIONS = [
  "Possible is tentative and treated as an unassigned prediction.",
  "Misses and processing errors are unsuccessful end-to-end outcomes.",
  "Cosine similarity is not a probability.",
];

export function evaluationReport(snapshot, data, outcome) {
  const { results, ...provenance } = snapshot;
  return {
    ...provenance, ground_truth_source: data.source, ground_truth_sha256: data.sourceHash ?? null,
    protocol: data.mode === "import" ? "Maximum cardinality, then maximum total IoU (>= 0.50)" : "Independent manual correspondence",
    scope: outcome.complete ? "Complete" : "Verified subset",
    approximate_boxes: Boolean(data.approximate),
    warnings: data.warnings ?? [],
    limitations: [...LIMITATIONS, ...(data.approximate ? ["Composite box locations are approximate."] : []),
      ...(!outcome.complete ? ["Unverified detections and unmarked missed faces may remain."] : [])],
    ...outcome,
  };
}

export function evaluationCsv(report) {
  const header = ["analysis_id", "image_filename", "image_sha256", "gallery_revision", "pipeline_id", "recognition_rules",
    "ground_truth_source", "ground_truth_sha256", "protocol", "scope", "limitations", "metric_numerators_and_denominators", "face_number",
    "prediction_status", "predicted_identity", "tentative_identity", "actual_face_id", "actual_identity",
    "actual_membership", "outcome", "detection_score", "cosine_similarity",
    "pred_x", "pred_y", "pred_w", "pred_h", "actual_x", "actual_y", "actual_w", "actual_h", "matched_iou"];
  const metrics = JSON.stringify({ recognition: report.metrics, detection: report.detection });
  const rows = report.rows.map(({ prediction: p, actual: a, outcome, iou }) => [
    report.analysis_id, report.filename, report.image_sha256, report.gallery_revision, report.pipeline_id,
    JSON.stringify(report.recognition_rules), report.ground_truth_source, report.ground_truth_sha256, report.protocol, report.scope,
    report.limitations.join(" "), metrics, p?.face_number, p?.status,
    p?.status === "recognised" ? p.identity : null, p?.status === "possible" ? p.identity : null,
    a?.id, a?.identity, a ? (a.known ? "known" : "unknown") : null, outcome, p?.confidence, p?.best_score,
    ...(p?.bbox ?? Array(4)), ...(a?.bbox ?? Array(4)), iou,
  ]);
  return [header, ...rows].map(row => row.map(csvCell).join(",")).join("\r\n") + "\r\n";
}
