import { apiRequest, setAuthStorage } from "./client.js";

export async function fetchVersion() {
  return apiRequest("/api/version");
}

export async function login(email, password) {
  const d = await apiRequest("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
  setAuthStorage({ accessToken: d.access_token });
  return d;
}

export async function register(email, password) {
  const d = await apiRequest("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
  setAuthStorage({ accessToken: d.access_token });
  return d;
}

export async function listTeams() {
  return apiRequest("/api/teams");
}

export async function createTeam(name) {
  return apiRequest("/api/teams", { method: "POST", body: JSON.stringify({ name }) });
}
