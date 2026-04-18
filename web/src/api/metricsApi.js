import { apiRequest } from "./client";

export function fetchMetricsList() {
  return apiRequest("/api/metrics");
}

export function fetchMetricsBySource(sourcePath) {
  return apiRequest(`/api/metrics?source=${encodeURIComponent(sourcePath)}`);
}
