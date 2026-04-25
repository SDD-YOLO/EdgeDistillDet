export async function invokeAgentStreamViaRelay(body) {
  return fetch("/api/agent/model/invoke-stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}
