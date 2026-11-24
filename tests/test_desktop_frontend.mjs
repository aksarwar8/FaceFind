import assert from "node:assert/strict";
import { test } from "node:test";
import { summaryEntries } from "../web/analysis.js";
import { recognitionCsv } from "../web/exports.js";
import { enrolledRows, identityRows, mountGallery } from "../web/gallery.js";
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

test("pending cards open previews by immutable ID and advance after confirmation", async t => {
  class Element {
    constructor(tag) {
      Object.assign(this, { tag, children: [], dataset: {}, style: {}, attributes: {},
        handlers: {}, classList: { toggle() {} }, value: "", hidden: false });
    }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener(name, listener) { this.handlers[name] = listener; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    focus() { this.focused = true; }
    click() { this.handlers.click?.(); }
    all(predicate) { return this.children.flatMap(child => [
      ...(predicate(child) ? [child] : []), ...child.all(predicate),
    ]); }
    querySelectorAll() { return this.all(el => Boolean(el.dataset.pendingId)); }
  }
  const nodes = new Map();
  const get = id => {
    if (!nodes.has(id)) nodes.set(id, new Element("div"));
    return nodes.get(id);
  };
  const replaceGlobal = (key, value) => {
    const before = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, { configurable: true, value });
    t.after(() => before ? Object.defineProperty(globalThis, key, before) : delete globalThis[key]);
  };
  replaceGlobal("document", {
    getElementById: get, createElement: tag => new Element(tag),
    createElementNS: (_, tag) => new Element(tag),
  });
  replaceGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  const pending = ["first", "second", "third"].map((id, index) => ({
    pending_id: `immutable-${id}`, identity: `Person ${index + 1}`, filename: `Person ${index + 1}.jpg`, faces: 2,
  }));
  const requested = [];
  let stale;
  replaceGlobal("fetch", async url => {
    requested.push(url);
    const item = pending.find(reference => url ===
      `/api/gallery/pending/${reference.pending_id}?interactive=true`);
    assert.ok(item);
    if (item.pending_id === stale) {
      return { ok: false, status: 409, json: async () => ({ detail: "Stale reference" }) };
    }
    const preview = "data:image/jpeg;base64,example";
    return { ok: true, json: async () => ({ ...item, width: 40, height: 40, preview,
      candidates: [0, 1].map(index => ({ index, confidence: .9, bbox: [0, 0, 10, 10], crop: preview })),
    }) };
  });
  get("identity-filter").value = "all";
  const state = { connected: true, workspace: "gallery", galleryFiles: [], status: {
    engine_initialized: true, gallery_loaded: true, active_pipeline_id: "scrfd_adaface",
    identities: ["Already enrolled"], people_count: 1, pending,
  } };
  let submission;
  function remove(item, accepted) {
    state.status = { ...state.status, people_count: state.status.people_count + Number(accepted),
      pending: state.status.pending.filter(r => r.pending_id !== item.pending_id) };
    render();
  }
  const render = mountGallery(state, { review(item, index) {
    submission = [item.pending_id, index];
    state.operation = { kind: "review" };
    remove(item, true);
    state.operation = null;
    render();
  }, refresh() { remove(pending[0], false); } });
  const settle = () => new Promise(resolve => setImmediate(resolve));
  const cards = () => get("identity-grid").querySelectorAll("button[data-pending-id]");
  const control = text => get("pending-item").all(el => el.tag === "button" && el.textContent === text)[0];
  render();
  await settle();
  assert.equal(get("pending-section").hidden, false);
  assert.match(requested[0], /immutable-first/);
  assert.equal(cards().length, 3);
  assert.ok(cards().every(card => card.tag === "button"));
  cards()[1].click();
  await settle();
  assert.match(requested.at(-1), /immutable-second/);
  assert.equal(cards()[1].attributes["aria-pressed"], "true");
  assert.equal(get("pending-section").focused, true);
  const boxes = get("pending-item").all(el => el.className === "pending-face-box");
  assert.equal(boxes.length, 2);
  boxes[1].click();
  control("Confirm selection").click();
  await settle();
  assert.deepEqual(submission, ["immutable-second", 1]);
  assert.match(requested.at(-1), /immutable-third/);
  assert.match(get("gallery-count").textContent, /2 enrolled/);
  assert.match(get("pending-position").textContent, /Reference 2 of 2 pending/);
  stale = "immutable-first";
  cards()[0].click();
  await settle();
  assert.equal(control("Refresh gallery").hidden, false);
  assert.equal(control("Confirm selection").disabled, true);
  control("Refresh gallery").click();
  await settle();
  assert.match(requested.at(-1), /immutable-third/);
  assert.equal(cards().length, 1);
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
