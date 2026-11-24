export const IOU_THRESHOLD = 0.5;
export const MAX_FACES = 256;
export const OUTCOMES = {
  correct_known: "Correct identity", wrong_known: "Wrong identity", rejected_known: "Incorrect rejection",
  correct_unknown: "Correct Unknown", false_accept: "False acceptance", missed_known: "Missed known face",
  missed_unknown: "Missed unknown face", processing_failure: "Processing failure",
  false_detection: "False detection", unverified: "Not verified",
};

export function analysisSnapshot(result) {
  const { annotated_image, ...data } = result;
  const snapshot = structuredClone(data);
  function freeze(value) {
    if (value && typeof value === "object") {
      for (const child of Object.values(value)) freeze(child);
      Object.freeze(value);
    }
    return value;
  }
  return freeze(snapshot);
}

export function currentEvaluation(snapshot, result, status) {
  return Boolean(result?.analysis_id && snapshot?.analysis_id === result.analysis_id
    && snapshot.gallery_revision === status?.gallery_revision
    && snapshot.pipeline_id === status?.active_pipeline_id);
}

export const identityKey = name => String(name ?? "").trim().normalize("NFC").toLocaleLowerCase("en");
export function galleryMap(snapshot) {
  return new Map(snapshot.gallery_identities.map(name => [identityKey(name), name]));
}

export function iou(a, b) {
  const area = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]))
    * Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const union = a[2] * a[3] + b[2] * b[3] - area;
  return union > 0 ? area / union : 0;
}

// Rectangular Hungarian assignment. A bonus greater than the total possible IoU
// makes cardinality the first objective. Extra columns allow unmatched predictions.
export function matchBoxes(predictions, truth, threshold = IOU_THRESHOLD) {
  const n = predictions.length;
  if (n > MAX_FACES || truth.length > MAX_FACES) throw new Error("IoU matching supports up to 256 detections and annotations.");
  const m = truth.length + n;
  const overlap = predictions.map(prediction => truth.map(face => iou(prediction.bbox, face.bbox)));
  const u = Array(n + 1).fill(0), v = Array(m + 1).fill(0);
  const assigned = Array(m + 1).fill(0), way = Array(m + 1).fill(0);
  const cost = (i, j) => {
    const value = overlap[i - 1][j - 1];
    return value >= threshold ? -(n + 1 + value) : 0;
  };
  for (let i = 1; i <= n && truth.length; i++) {
    assigned[0] = i;
    let j = 0;
    const minimum = Array(m + 1).fill(Infinity), used = Array(m + 1).fill(false);
    do {
      used[j] = true;
      const row = assigned[j];
      let delta = Infinity, next = 0;
      for (let k = 1; k <= m; k++) {
        if (used[k]) continue;
        const reduced = cost(row, k) - u[row] - v[k];
        if (reduced < minimum[k]) { minimum[k] = reduced; way[k] = j; }
        if (minimum[k] < delta) { delta = minimum[k]; next = k; }
      }
      for (let k = 0; k <= m; k++) {
        if (used[k]) { u[assigned[k]] += delta; v[k] -= delta; }
        else minimum[k] -= delta;
      }
      j = next;
    } while (assigned[j]);
    do {
      const previous = way[j];
      assigned[j] = assigned[previous];
      j = previous;
    } while (j);
  }
  return new Map(assigned.slice(1, truth.length + 1).flatMap((row, index) =>
    row && overlap[row - 1][index] >= threshold ? [[row - 1, { index, iou: overlap[row - 1][index] }]] : []));
}

function classify(prediction, actual) {
  if (!prediction) return actual.known ? "missed_known" : "missed_unknown";
  if (prediction.status === "error" || !["recognised", "possible", "unknown"].includes(prediction.status)
      || prediction.status === "recognised" && !prediction.identity) return "processing_failure";
  const accepted = prediction.status === "recognised";
  if (!actual.known) return accepted ? "false_accept" : "correct_unknown";
  if (!accepted) return "rejected_known";
  return identityKey(prediction.identity) === identityKey(actual.identity) ? "correct_known" : "wrong_known";
}

export function evaluate(snapshot, data) {
  const predictions = snapshot.results;
  const imported = data.mode === "import";
  const complete = imported || Boolean(data.complete && predictions.every(face => data.labels.has(face.face_number)));
  const rows = [];
  const truth = imported ? data.faces : [];
  const matches = imported ? matchBoxes(predictions, truth) : new Map();
  const matched = new Set();
  function append(prediction, actual, result, overlap = null) {
    rows.push({ key: prediction ? `p-${prediction.face_number}` : `g-${actual.id}`,
      prediction, actual, outcome: result, iou: overlap });
  }
  predictions.forEach((prediction, index) => {
    const match = matches.get(index);
    const actual = imported ? (match && truth[match.index]) : data.labels.get(prediction.face_number);
    if (match) matched.add(match.index);
    if (actual && !actual.falseDetection) append(prediction, actual, classify(prediction, actual), match?.iou);
    else append(prediction, null, imported || actual?.falseDetection ? "false_detection" : "unverified");
  });
  for (const actual of imported ? truth.filter((_, index) => !matched.has(index)) : data.missed) {
    append(null, actual, classify(null, actual));
  }
  const counts = Object.fromEntries(Object.keys(OUTCOMES).map(key => [key, 0]));
  const matrix = { known: [0, 0, 0, 0, 0], unknown: [null, 0, 0, 0, 0] };
  let known = 0, unknown = 0, validKnown = 0, validUnknown = 0, matchesCount = 0;
  for (const { prediction, actual, outcome } of rows) {
    counts[outcome]++;
    if (!actual) continue;
    if (prediction) matchesCount++;
    const column = ({ correct_known: 0, wrong_known: 1, false_accept: 1, rejected_known: 2,
      correct_unknown: 2, missed_known: 3, missed_unknown: 3, processing_failure: 4 })[outcome];
    matrix[actual.known ? "known" : "unknown"][column]++;
    if (actual.known) known++;
    else unknown++;
    if (prediction && outcome !== "processing_failure") {
      if (actual.known) validKnown++;
      else validUnknown++;
    }
  }
  const ratio = (numerator, denominator) => ({ numerator, denominator, value: denominator ? numerator / denominator : null });
  const c = counts;
  const missed = c.missed_known + c.missed_unknown;
  const fp = c.false_detection;
  const detection = complete ? {
    method: imported ? "iou" : "manual",
    tp: matchesCount, fp, fn: missed,
    precision: ratio(matchesCount, matchesCount + fp),
    recall: ratio(matchesCount, known + unknown),
    f1: ratio(2 * matchesCount, 2 * matchesCount + fp + missed),
  } : null;
  return {
    rows, counts, matrix, complete, detection, known, unknown,
    detected: predictions.length, verified: predictions.length - c.unverified,
    metrics: {
      known: ratio(c.correct_known, known),
      unknown: ratio(c.correct_unknown, unknown),
      falseAcceptance: ratio(c.false_accept, unknown),
      knownFailure: ratio(known - c.correct_known, known),
      overall: ratio(c.correct_known + c.correct_unknown, known + unknown),
      conditionalKnown: ratio(c.correct_known, validKnown),
      conditionalUnknown: ratio(c.correct_unknown, validUnknown),
    },
  };
}
