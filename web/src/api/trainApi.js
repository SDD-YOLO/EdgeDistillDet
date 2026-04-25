import { apiRequest } from "./client";

export function fetchResumeCandidates(project) {
  const params = new URLSearchParams({ project });
  return apiRequest(`/api/train/resume_candidates?${params.toString()}`);
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

export async function downloadTrainLogsBlob() {
  const response = await fetch("/api/train/logs/download");
  if (!response.ok) {
    throw new Error("下载日志失败");
  }
  return response.blob();
}
