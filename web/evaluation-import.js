import { galleryMap, identityKey, MAX_FACES } from "./evaluation-core.js";

export const MAX_TRUTH_BYTES = 2 * 1024 * 1024;
export const TEMPLATE = "image_filename,face_id,actual_identity,status,bbox_x,bbox_y,bbox_width,bbox_height,image_width,image_height,coordinate_space\r\n";
const BOX = ["bbox_x", "bbox_y", "bbox_width", "bbox_height"];
const DEMO = ["group", "face_number", "source_identity"];

function csvRows(text) {
  const rows = [];
  let row = [], value = "", quoted = false, closed = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { value += '"'; i++; }
      else if (c === '"') { quoted = false; closed = true; }
      else value += c;
    } else if (c === "," || c === "\n" || c === "\r") {
      row.push(value); value = ""; closed = false;
      if (c !== ",") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        if (row.some(Boolean)) rows.push(row);
        row = [];
      }
    } else if (closed || c === '"' && value) throw new Error("Invalid CSV quoting.");
    else if (c === '"') quoted = true;
    else value += c;
  }
  if (quoted) throw new Error("Unclosed CSV quote.");
  if (value || row.length) rows.push([...row, value]);
  const header = rows.shift()?.map(field => field.trim());
  if (!header?.length || new Set(header).size !== header.length) throw new Error("Missing or duplicate CSV columns.");
  if (rows.some(fields => fields.length !== header.length)) throw new Error("CSV rows must match the header.");
  return { header, records: rows.map(fields => Object.fromEntries(header.map((field, index) => [field, fields[index]]))) };
}

// Coordinates represent pixel edges. Transform all four corners before scaling.
export function mapBox(box, width, height, orientation, workingWidth, workingHeight) {
  const [x, y, w, h] = box;
  if (![x, y, w, h, width, height, workingWidth, workingHeight].every(Number.isFinite)
    || width <= 0 || height <= 0 || workingWidth <= 0 || workingHeight <= 0 || x < 0 || y < 0
    || w <= 0 || h <= 0 || x + w > width + 1e-7 || y + h > height + 1e-7) {
    throw new Error("Bounding box is invalid or outside its coordinate dimensions.");
  }
  const transform = (a, b) => [
    [a, b], [width-a, b], [width-a, height-b], [a, height-b],
    [b, a], [height-b, a], [height-b, width-a], [b, width-a],
  ][orientation - 1];
  if (!Number.isInteger(orientation) || orientation < 1 || orientation > 8) throw new Error("Invalid image orientation.");
  const points = [transform(x, y), transform(x+w, y), transform(x, y+h), transform(x+w, y+h)];
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const sx = workingWidth / (orientation >= 5 ? height : width);
  const sy = workingHeight / (orientation >= 5 ? width : height);
  return [Math.min(...xs)*sx, Math.min(...ys)*sy, (Math.max(...xs)-Math.min(...xs))*sx, (Math.max(...ys)-Math.min(...ys))*sy];
}

function numeric(value) {
  if (typeof value !== "number" && (typeof value !== "string" || !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim()))) {
    throw new Error("Coordinates and dimensions must be finite numbers.");
  }
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error("Coordinates and dimensions must be finite numbers.");
  return number;
}
const filenameKey = name => String(name ?? "").trim().normalize("NFC").toLowerCase();
const missing = value => value == null || value === "";
const identifier = value => (typeof value === "number" && Number.isSafeInteger(value)
  || typeof value === "string" && value.trim()) ? String(value).trim() : "";
function checkedFilename(name) {
  if (!identifier(name) || typeof name !== "string" || /[\\/:\x00-\x1f]/.test(name)) {
    throw new Error("Use an image basename or a separate SHA-256 identifier.");
  }
  return name.trim();
}

function geometry(row, image, snapshot) {
  const space = row.coordinate_space || image?.coordinate_space || "oriented";
  if (!["oriented", "raw", "working"].includes(space)) throw new Error("Unsupported coordinate space.");
  const raw = snapshot.image_geometry;
  const orientation = space === "raw" ? raw?.orientation : 1;
  if (!orientation) throw new Error("Raw image orientation is unavailable for this analysis.");
  const swap = space === "oriented" && raw?.orientation >= 5;
  const sw = space === "working" ? snapshot.image_width : swap ? raw.original_height : raw?.original_width;
  const sh = space === "working" ? snapshot.image_height : swap ? raw.original_width : raw?.original_height;
  const w = row.image_width ?? image?.image_width;
  const h = row.image_height ?? image?.image_height;
  if (missing(w) !== missing(h)) throw new Error("Supply both coordinate dimensions.");
  const width = missing(w) ? sw ?? snapshot.image_width : numeric(w);
  const height = missing(h) ? sh ?? snapshot.image_height : numeric(h);
  if (![width, height].every(v => Number.isSafeInteger(v) && v > 0 && v <= 100000)
      || sw && sh && Math.abs(width / height / (sw / sh) - 1) > .02) {
    throw new Error("Invalid coordinate dimensions or image aspect ratio mismatch.");
  }
  return [width, height, orientation, space];
}

export function parseGroundTruth(text, extension, snapshot) {
  if (new TextEncoder().encode(text).length > MAX_TRUTH_BYTES) throw new Error("Ground truth exceeds 2 MiB.");
  text = text.replace(/^\uFEFF/, "");
  let records, manifestGroups, approximate = false;
  if (extension === "csv") {
    const { header, records: items } = csvRows(text);
    const demo = DEMO.every(name => header.includes(name));
    const canonical = ["face_id", "actual_identity"].every(name => header.includes(name))
      && ["image_filename", "image_sha256", "image_id"].some(name => header.includes(name));
    if (!demo && !canonical) throw new Error("Missing image identifier, face ID or actual identity columns.");
    if (BOX.some(name => header.includes(name)) && !BOX.every(name => header.includes(name))) {
      throw new Error("Supply all four bounding-box columns.");
    }
    records = items;
    approximate = demo;
  } else if (extension === "json") {
    const data = JSON.parse(text);
    records = Array.isArray(data) ? data : data?.faces;
    if (!Array.isArray(records)) throw new Error("Use a JSON face array or an object with a faces array.");
    if (records.some(row => !row || typeof row !== "object" || Array.isArray(row))) throw new Error("Invalid face record.");
    manifestGroups = data?.groups;
    approximate = Boolean(data?.bounding_box_method || data?.box_quality === "approximate");
  } else throw new Error("Choose a CSV or JSON ground-truth file.");
  if (!records.length) throw new Error("Ground truth has no annotations.");
  let image, matchingGroups;
  if (manifestGroups !== undefined) {
    if (!Array.isArray(manifestGroups)) throw new Error("Invalid group manifest.");
    matchingGroups = manifestGroups.filter(g => filenameKey(g?.image) === filenameKey(snapshot.filename));
    if (matchingGroups.length > 1) throw new Error("Ambiguous filename in the group manifest.");
    const group = matchingGroups[0];
    if (group) {
      if (group.sha256 && group.sha256.toLowerCase() !== snapshot.image_sha256) throw new Error("Image SHA-256 disagrees with the manifest.");
      image = { image_width: group.canvas_width, image_height: group.canvas_height, expected: group.expected_detection_count };
    }
  }
  const spellings = new Set(), hashes = new Set();
  const relevant = records.filter(row => {
    const name = row.image_filename ?? row.group;
    const hash = row.image_sha256 ?? row.image_id;
    if (missing(name) && missing(hash)) throw new Error("Every row needs an image filename or SHA-256.");
    const matchesName = !missing(name) && filenameKey(checkedFilename(name)) === filenameKey(snapshot.filename);
    if (!missing(hash) && (typeof hash !== "string" || !/^[0-9a-f]{64}$/i.test(hash))) throw new Error("Invalid image SHA-256.");
    const matchesHash = !missing(hash) && hash.toLowerCase() === snapshot.image_sha256;
    if (matchesName) {
      spellings.add(name);
      if (hash) hashes.add(hash.toLowerCase());
      if (hash && !matchesHash) throw new Error("Image SHA-256 disagrees with the analyzed photo.");
    }
    if (matchesHash && name) spellings.add(name);
    return matchesName || matchesHash;
  });
  if (spellings.size > 1 || hashes.size > 1) throw new Error("Ambiguous filename association.");
  if (!relevant.length) throw new Error("No ground truth belongs to this image.");
  if (relevant.length > MAX_FACES) throw new Error("A group may contain up to 256 ground-truth faces.");
  if (image?.expected != null && image.expected !== relevant.length) throw new Error("Group count disagrees with its listed annotations.");

  const gallery = galleryMap(snapshot), ids = new Set(), warnings = [];
  const faces = relevant.map(row => {
    const id = identifier(row.face_id ?? row.face_number);
    if (!id || ids.has(id)) throw new Error("Missing or duplicate ground-truth face ID.");
    ids.add(id);
    const name = row.actual_identity ?? row.source_identity;
    if (typeof name !== "string" || !name.trim() || name.length > 240 || /[\x00-\x1f]/.test(name)) throw new Error("Invalid actual identity.");
    const identity = gallery.get(identityKey(name)) ?? name.trim();
    const known = gallery.has(identityKey(name));
    const supplied = String(row.status ?? "").trim().toLowerCase();
    if (supplied && !["known", "unknown"].includes(supplied)) throw new Error("Expected known or unknown annotation status.");
    if (supplied && (supplied === "known") !== known) warnings.push(`Face ${id}: supplied ${supplied}; gallery says ${known ? "known" : "unknown"}.`);
    const boxes = row.bbox == null ? BOX.map(key => row[key]) : row.bbox;
    if (!Array.isArray(boxes) || boxes.length !== 4 || boxes.some(missing) && !boxes.every(missing)) {
      throw new Error("Incomplete or invalid bounding box.");
    }
    const [width, height, orientation, space] = geometry(row, image, snapshot);
    const box = boxes.every(missing) ? null : boxes.map(numeric);
    const bbox = box && mapBox(box, width, height, orientation, snapshot.image_width, snapshot.image_height);
    const reliability = String(row.box_reliable ?? "").trim().toLowerCase();
    if (reliability && !["true", "false"].includes(reliability)) throw new Error("Box reliability must be true or false.");
    return { id, identity, known, bbox: reliability === "false" ? null : bbox, space };
  });
  if (faces.some(face => face.bbox) && new Set(faces.map(face => face.space)).size > 1) {
    throw new Error("Use one coordinate space per image.");
  }
  faces.sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
  return { faces, warnings, approximate, manualRequired: faces.some(face => !face.bbox) };
}
