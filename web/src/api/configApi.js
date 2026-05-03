import { apiRequest } from "./client";

export function pickDialogPath({
  kind = "file",
  title = "选择路径",
  initial_path = "",
  filters = [],
} = {}) {
  return apiRequest("/api/dialog/pick", {
    method: "POST",
    body: JSON.stringify({
      kind,
      title,
      initial_path,
      filters,
    }),
  });
}

export function fetchDistillConfig() {
  return apiRequest("/api/config/distill_config.yaml");
}

export function checkOutputPath(project) {
  const params = new URLSearchParams({ project });
  return apiRequest(`/api/output/check?${params.toString()}`);
}

export function saveDistillConfig(config) {
  return apiRequest("/api/config/save", {
    method: "POST",
    body: JSON.stringify({ name: "distill_config.yaml", config }),
  });
}

export function uploadConfigFile({ content, name }) {
  return apiRequest("/api/config/upload", {
    method: "POST",
    body: JSON.stringify({ content, name }),
  });
}

export function fetchRecentConfig() {
  return apiRequest("/api/config/recent");
}
