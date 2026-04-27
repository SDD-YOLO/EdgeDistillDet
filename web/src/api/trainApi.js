import { apiRequest } from "./client";

export function fetchResumeCandidates(project) {
  const params = new URLSearchParams({ project });
  return apiRequest(`/api/train/resume_candidates?${params.toString()}`);
}

export function fetchExportWeightCandidates(project) {
  const params = new URLSearchParams({ project });
  return apiRequest(`/api/train/export_weight_candidates?${params.toString()}`);
}

export function fetchTrainStatus() {
  return apiRequest("/api/train/status");
}

export function fetchTrainLogs({ offset = 0, limit = 120 } = {}) {
  return apiRequest(`/api/train/logs?offset=${offset}&limit=${limit}`);
}

export function startTrain(body) {
  return apiRequest("/api/train/start", { method: "POST", body: JSON.stringify(body) });
}

export function stopTrain() {
  return apiRequest("/api/train/stop", { method: "POST", body: JSON.stringify({}) });
}

export function startDisplay(body) {
  return apiRequest("/api/display/start", { method: "POST", body: JSON.stringify(body) });
}

export function stopDisplay() {
  return apiRequest("/api/display/stop", { method: "POST", body: JSON.stringify({}) });
}

export function fetchDisplayStatus() {
  return apiRequest("/api/display/status");
}

export function fetchDisplayLogs({ offset = 0, limit = 120 } = {}) {
  const params = new URLSearchParams({ offset, limit });
  return apiRequest(`/api/display/logs?${params.toString()}`);
}

export function startExportModel(body) {
  return apiRequest("/api/export/start", { method: "POST", body: JSON.stringify(body) });
}

export function stopExportModel() {
  return apiRequest("/api/export/stop", { method: "POST", body: JSON.stringify({}) });
}

export function fetchExportStatus() {
  return apiRequest("/api/export/status");
}

export function fetchExportLogs({ offset = 0, limit = 120 } = {}) {
  const params = new URLSearchParams({ offset, limit });
  return apiRequest(`/api/export/logs?${params.toString()}`);
}

export async function downloadTrainLogsBlob() {
  const response = await fetch("/api/train/logs/download");
  if (!response.ok) {
    throw new Error("下载日志失败");
  }
  return response.blob();
}
