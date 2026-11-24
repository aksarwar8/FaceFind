import assert from "node:assert/strict";
import { test } from "node:test";
import { summaryEntries } from "../web/analysis.js";
import { recognitionCsv } from "../web/exports.js";
import { enrolledRows, identityRows } from "../web/gallery.js";
import { candidateLabel, validateReferencePreview } from "../web/reference-review.js";
import { referenceSelectionPolicy } from "../web/settings.js";

test("the saved reference policy can opt out of automatic selection", t => {
  let stored;
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true, value: { getItem: key => key === "facefind.referenceSelectionPolicy" ? stored : null },
  });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, "localStorage", descriptor);
    else delete globalThis.localStorage;
  });
  assert.equal(referenceSelectionPolicy(), "smart");
  stored = "always_review";
  assert.equal(referenceSelectionPolicy(), "always_review");
  stored = "unrecognized";
  assert.equal(referenceSelectionPolicy(), "smart");
});

test("reference review labels ordered candidates without changing backend indices", () => {
  const jpeg = "data:image/jpeg;base64,example";
  const item = { pending_id: "immutable-review-id", identity: "Sample", filename: "Sample.jpg", faces: 2 };
  const first = { index: 0, confidence: 0.829, bbox: [4, 5, 20, 20], crop: jpeg };
  const second = { index: 1, confidence: 0.533, bbox: [25, 5, 6, 6], crop: null };
  const response = { ...item, width: 40, height: 30, preview: jpeg, candidates: [first, second] };
  assert.deepEqual(validateReferencePreview(response, item).candidates.map(face => [face.index, candidateLabel(face.index)]),
    [[0, "Face 1"], [1, "Face 2"]]);
  assert.throws(() => validateReferencePreview({ ...response, pending_id: "replacement-id" }, item), /preview changed/);
  assert.throws(() => validateReferencePreview({ ...response, candidates: [second, first] }, item), /preview changed/);
});

test("a fresh Gallery view uses the latest replacement errors", () => {
  const status = {
    identities: ["Enrolled sample"],
    pending: [],
    gallery_build_errors: ["replacement.txt: Unsupported file format"],
    gallery_timings: {
      per_reference: [{ identity: "older.jpg", status: "rejected", error: "No face detected" }],
    },
  };
  assert.deepEqual(identityRows({ status }).map(row => [row.name, row.status, row.label]), [
    ["Enrolled sample", "accepted", "Enrolled"],
    ["replacement.txt", "rejected", "Unsupported file format"],
  ]);
});

test("gallery and dashboard previews share revision-scoped thumbnail URLs", () => {
  const revision = "a".repeat(32);
  const rows = enrolledRows({ identities: ["A", "B"], thumbnail_indices: [1], gallery_revision: revision });
  assert.deepEqual(rows.map(row => row.thumbnail), [null, `/api/gallery/thumbnail/${revision}/1`]);
});

test("the analysis summary respects pipeline-specific statuses and processing errors", () => {
  const result = { face_count: 4, summary: { recognised: 2, unknown: 1, possible: 0, errors: 1 } };
  const entries = pipeline_id => Object.fromEntries(summaryEntries({ ...result, pipeline_id }));
  assert.deepEqual(entries("scrfd_adaface"), { Detected: 4, Recognized: 2, Unknown: 1, "Processing errors": 1 });
  assert.equal(entries("yunet_sface")["Possible matches"], 0);
});

test("CSV prevents spreadsheet formulas without turning similarity scores into text", () => {
  const output = recognitionCsv([
    { face_number: 1, status: "recognised", identity: ' =HYPERLINK("a,b")', best_score: 0.495 },
    { face_number: 2, status: "possible", identity: "+candidate", best_score: -0.1, conflict_with: 1 },
    { face_number: 3, status: "unknown", identity: "Should stay hidden", best_score: 0.2 },
    { face_number: 4, status: "error", error: "line one, then\nline two" },
  ]);
  assert.ok(output.includes(`"' =HYPERLINK(""a,b"")"`));
  assert.ok(output.includes(`"possible","","'+candidate","-0.1","1"`));
  assert.ok(!output.includes("Should stay hidden"));
  assert.ok(output.includes('"line one, then\nline two"'));
  assert.ok(output.endsWith("\r\n"));
});
