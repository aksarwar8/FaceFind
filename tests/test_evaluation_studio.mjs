import assert from "node:assert/strict";
import { test } from "node:test";
import { analysisSnapshot, currentEvaluation, evaluate, matchBoxes } from "../web/evaluation-core.js";
import { mapBox, parseGroundTruth, TEMPLATE } from "../web/evaluation-import.js";
import { evaluationCsv, evaluationReport } from "../web/evaluation-export.js";
import { filteredRows, formatMetric } from "../web/evaluation-view.js";

const box = n => [n * 10, 0, 8, 8];
const p = (n, status, identity = null) => ({ face_number: n, status, identity, bbox: box(n), confidence: .81, best_score: .42 });
const gt = (n, identity, known) => ({ id: String(n), identity, known, bbox: box(n) });
const snapshot = (results = []) => ({
  analysis_id: "analysis", filename: "photo.jpg", image_sha256: "a".repeat(64), pipeline_id: "scrfd_adaface",
  gallery_revision: "gallery", gallery_identities: ["Alice", "Bob", "=formula"], image_width: 100, image_height: 50,
  image_geometry: { original_width: 200, original_height: 100, orientation: 1 }, results,
});
const imported = faces => ({ mode: "import", faces, source: "annotations.csv" });

test("detection and recognition outcomes use separate assessed denominators", () => {
  const result = evaluate(snapshot([
    p(1, "recognised", "Alice"), p(2, "recognised", "Alice"), p(3, "unknown"),
    p(4, "unknown"), p(5, "recognised", "Bob"), p(6, "error"), p(9, "recognised", "Bob"),
  ]), imported([
    gt(1, "Alice", true), gt(2, "Bob", true), gt(3, "Alice", true),
    gt(4, "Stranger", false), gt(5, "Stranger", false), gt(6, "Bob", true),
    gt(7, "Alice", true), gt(8, "Stranger", false),
  ]));
  assert.deepEqual(result.counts, {
    correct_known: 1, wrong_known: 1, rejected_known: 1, correct_unknown: 1, false_accept: 1,
    missed_known: 1, missed_unknown: 1, processing_failure: 1, false_detection: 1, unverified: 0,
  });
  assert.deepEqual(result.matrix, { known: [1, 1, 1, 1, 1], unknown: [null, 1, 1, 1, 0] });
  assert.deepEqual([result.detection.tp, result.detection.fp, result.detection.fn], [6, 1, 2]);
  assert.equal(result.detection.f1.value, 12 / 15);
  assert.equal(result.metrics.overall.value, 2 / 8);
  assert.equal(result.metrics.knownFailure.value, 4 / 5);
  assert.equal(result.metrics.conditionalKnown.denominator, 3);
  assert.equal(result.metrics.conditionalUnknown.denominator, 2);
  assert.deepEqual(filteredRows(result.rows, "missed").map(row => row.actual.id), ["7", "8"]);
  assert.equal(filteredRows(result.rows, "false_detection")[0].prediction.identity, "Bob");
});

test("IoU optimizes cardinality before IoU, then prefers the highest total overlap", () => {
  const make = (...boxes) => boxes.map(bbox => ({ bbox }));
  const targets = make([0, 0, 10, 10], [5, 0, 10, 10]);
  const tricky = make([2, 0, 10, 10], [0, 0, 10, 10]);
  assert.deepEqual([...matchBoxes(tricky, targets)].sort().map(([i, m]) => [i, m.index]), [[0, 1], [1, 0]]);
  const duplicates = make([1, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]);
  const best = matchBoxes(duplicates, targets.slice(0, 1));
  assert.equal(best.size, 1);
  assert.equal(best.values().next().value.iou, 1);
  assert.deepEqual([...matchBoxes(duplicates, targets.slice(0, 1))], [...best]);
});

test("perfect, all known, all unknown, Possible, and empty denominators", () => {
  const known = evaluate(snapshot([p(1, "recognised", "Alice")]), imported([gt(1, "Alice", true)]));
  assert.equal(known.metrics.overall.value, 1);
  assert.equal(known.metrics.unknown.value, null);
  assert.equal(formatMetric(known.metrics.unknown), "N/A (0/0)");
  const unknown = evaluate(snapshot([p(1, "possible", "Alice")]), imported([gt(1, "Other", false)]));
  assert.equal(unknown.metrics.unknown.value, 1);
  assert.equal(unknown.metrics.known.value, null);
  const zero = evaluate(snapshot(), { mode: "manual", labels: new Map(), missed: [], complete: true });
  assert.equal(zero.metrics.overall.value, null);
});

test("oriented coordinates, downsampling and EXIF transforms preserve box edges", () => {
  assert.deepEqual(mapBox([10, 20, 30, 40], 200, 100, 6, 50, 100), [20, 5, 20, 15]);
  assert.deepEqual(mapBox([10, 20, 30, 40], 200, 100, 2, 100, 50), [80, 10, 15, 20]);
  assert.deepEqual(mapBox([10, 20, 30, 40], 200, 100, 8, 50, 100), [10, 80, 20, 15]);
  assert.throws(() => mapBox([0, 0, 0, 1], 200, 100, 1, 100, 50), /invalid/);
});

test("reads selected image rows, checks gallery membership, and maps composite metadata", () => {
  const csv = "group,face_number,source_identity,expected_label,status,bbox_x,bbox_y,bbox_width,bbox_height\r\n"
    + 'photo.jpg,1,"A, B",Unknown,known,20,20,20,20\r\nother.jpg,1,Bob,Bob,known,20,20,20,20';
  const data = parseGroundTruth(csv, "csv", snapshot());
  assert.equal(data.faces.length, 1);
  assert.deepEqual(data.faces[0].bbox, [10, 10, 10, 10]);
  assert.equal(data.faces[0].known, false);
  assert.match(data.warnings[0], /gallery says unknown/);
  const manifest = { groups: [{ image: "photo.jpg", sha256: "a".repeat(64), canvas_width: 200, canvas_height: 100,
    expected_detection_count: 1 }], bounding_box_method: "estimated",
    faces: [{ group: "photo.jpg", face_number: 1, source_identity: " Alice ", status: "unknown",
      bbox_x: 20, bbox_y: 20, bbox_width: 20, bbox_height: 20 }] };
  const fromJson = parseGroundTruth(JSON.stringify(manifest), "json", snapshot());
  assert.deepEqual(fromJson.faces[0].bbox, [10, 10, 10, 10]);
  assert.equal(fromJson.faces[0].identity, "Alice");
  assert.equal(fromJson.approximate, true);
  manifest.groups[0].sha256 = "b".repeat(64);
  assert.throws(() => parseGroundTruth(JSON.stringify(manifest), "json", snapshot()), /SHA-256/);
});

test("rejects empty, invalid, duplicate, ambiguous, and oversized annotation files", () => {
  const add = (...lines) => TEMPLATE + lines.join("\n");
  const row = "photo.jpg,1,Alice,known,10,10,20,20,200,100,oriented";
  for (const [csv, message] of [
    ["face_id,actual_identity\n1,Alice", /Missing/],
    [TEMPLATE, /no annotations/], [add(row, row), /duplicate/],
    [add(row, row.replace("photo.jpg,1", "Photo.jpg,2")), /Ambiguous/],
    [add(row.replace("photo.jpg", "other.jpg")), /belongs/],
    [add(row.replace("10,10", "Infinity,10")), /finite/],
    [add(row.replace("200,100", "200,1")), /dimensions/],
    [add(row.replace("10,10", "-1,10")), /outside/],
    [add(row.replace("20,20", "20,")), /Incomplete/],
    ["x".repeat(2 * 1024 * 1024 + 1), /2 MiB/],
  ]) assert.throws(() => parseGroundTruth(csv, "csv", snapshot()), message);
  const boxless = parseGroundTruth(JSON.stringify([{ image_filename: "photo.jpg", face_id: "01", actual_identity: "Alice" }]), "json", snapshot());
  assert.equal(boxless.manualRequired, true);
});

test("manual subsets, stale snapshots and CSV/JSON export preserve original predictions", () => {
  const source = snapshot([p(1, "recognised", "=formula"), p(2, "unknown")]);
  const frozen = analysisSnapshot(source);
  const data = { mode: "manual", source: "Manual review", labels: new Map([[1, gt(1, "=formula", true)]]), missed: [], complete: true };
  const partial = evaluate(frozen, data);
  assert.equal(partial.complete, false);
  assert.equal(partial.verified, 1);
  assert.equal(partial.detection, null);
  assert.equal(partial.metrics.overall.denominator, 1);
  assert.throws(() => { frozen.results[0].identity = "Changed"; }, TypeError);
  assert.equal(source.results[0].identity, "=formula");
  const status = { gallery_revision: "gallery", active_pipeline_id: "scrfd_adaface" };
  assert.equal(currentEvaluation(frozen, source, status), true);
  assert.equal(currentEvaluation(frozen, { ...source, analysis_id: "new" }, status), false);
  assert.equal(currentEvaluation(frozen, source, { ...status, gallery_revision: "changed" }), false);
  assert.equal(currentEvaluation(frozen, source, { ...status, active_pipeline_id: "yunet_sface" }), false);
  const report = evaluationReport(frozen, data, partial);
  assert.equal(report.scope, "Verified subset");
  assert.equal(report.image_sha256, source.image_sha256);
  assert.match(evaluationCsv(report), /"'=formula"/);
  assert.match(evaluationCsv(report), /metric_numerators_and_denominators/);
});
