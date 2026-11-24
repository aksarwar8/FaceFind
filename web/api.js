const API = "/api";

export async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(API + path, { cache: "no-store", ...options });
  } catch {
    throw Object.assign(new Error("Cannot reach the local FaceFind backend."), { disconnected: true });
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const message = typeof detail === "string" ? detail
      : Array.isArray(detail) ? detail.map(item => item.msg).filter(Boolean).join("; ")
      : "";
    throw Object.assign(new Error(message || `Request failed (HTTP ${response.status}).`), { status: response.status });
  }
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("The local backend returned an invalid response.");
  }
  return data;
}

export function post(path, body, status) {
  const headers = {
    "X-FaceFind-Pipeline": status?.active_pipeline_id ?? "",
    "X-FaceFind-Gallery": status?.gallery_revision ?? "",
  };
  if (body && !(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }
  return request(path, { method: "POST", headers, body });
}

export function jpeg(value) {
  return typeof value === "string" && value.startsWith("data:image/jpeg;base64,");
}

export function validStatus(data) {
  const policy = data.image_upload;
  if (typeof data.engine_initialized !== "boolean" || typeof data.gallery_loaded !== "boolean"
      || !Number.isInteger(data.people_count) || !Array.isArray(data.pending) || !Array.isArray(data.identities)
      || policy && (!Number.isSafeInteger(policy.max_bytes) || policy.max_bytes < 1
        || !Array.isArray(policy.formats) || !policy.formats.every(format => typeof format === "string"))) {
    throw new Error("The backend returned invalid gallery status.");
  }
  return data;
}
