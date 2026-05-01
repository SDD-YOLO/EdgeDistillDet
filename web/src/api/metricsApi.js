import { apiRequest } from "./client";

export function fetchMetricsList(options = {}) {
  return apiRequest("/api/metrics", options);
}

export function fetchMetricsBySource(sourcePath, options = {}) {
  return apiRequest(`/api/metrics?source=${encodeURIComponent(sourcePath)}`, options);
}
