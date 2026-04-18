import { apiRequest } from "./client";

export function fetchAgentTools() {
  return apiRequest("/api/agent/tools");
}

export function executeAgentTool({ tool, args = {} }) {
  return apiRequest("/api/agent/tools/execute", {
    method: "POST",
    body: JSON.stringify({ tool, args })
  });
}

export function previewAgentPatch(body) {
  return apiRequest("/api/agent/patch/preview", {
    method: "POST",
    body: JSON.stringify(body)
  });
}

export function fetchAgentConfigSchema() {
  return apiRequest("/api/agent/config-schema");
}
