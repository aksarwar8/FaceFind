export function csvCell(value) {
  const text = String(value ?? "");
  const dangerous = typeof value === "string" && /^[\s\uFEFF\u200B]*[=+\-@\t\r\n＝＋－＠]/u.test(text);
  return '"' + (dangerous ? "'" + text : text).replaceAll('"', '""') + '"';
}

export function recognitionCsv(results) {
  const header = ["Face number", "Status", "Assigned identity", "Candidate identity",
    "Cosine similarity", "Duplicate retained face", "Processing error"];
  const rows = results.map(face => [
    face.face_number, face.status,
    face.status === "recognised" ? face.identity : "",
    face.status === "possible" ? face.identity : "",
    Number.isFinite(face.best_score) ? face.best_score : "",
    face.conflict_with, face.error,
  ]);
  return [header, ...rows].map(row => row.map(csvCell).join(",")).join("\r\n") + "\r\n";
}

export function download(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  download(url, filename);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function exportAnalysis(analysis, format) {
  const name = (analysis.filename || "facefind").replace(/\.[^.]+$/, "").replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_");
  if (format === "jpeg") {
    download(analysis.annotated_image, `${name}-annotated.jpg`);
    return;
  }
  const url = URL.createObjectURL(new Blob(["\uFEFF", recognitionCsv(analysis.results)], { type: "text/csv;charset=utf-8" }));
  download(url, `${name}-recognition.csv`);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
