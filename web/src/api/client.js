const TOKEN_KEY = "edgedistilldet-access-token";
const TEAM_KEY = "edgedistilldet-team-id";

export function getStoredToken() {
  return window.localStorage.getItem(TOKEN_KEY) || "";
}

export function getStoredTeamId() {
  return window.localStorage.getItem(TEAM_KEY) || "";
}

export function setAuthStorage({ accessToken, teamId }) {
  if (accessToken !== undefined) {
    if (accessToken) window.localStorage.setItem(TOKEN_KEY, accessToken);
    else window.localStorage.removeItem(TOKEN_KEY);
  }
  if (teamId !== undefined) {
    if (teamId) window.localStorage.setItem(TEAM_KEY, teamId);
    else window.localStorage.removeItem(TEAM_KEY);
  }
}

export function authHeaders(extra = {}) {
  const h = { ...extra };
  const t = getStoredToken();
  const team = getStoredTeamId();
  if (t) h.Authorization = `Bearer ${t}`;
  if (team) h["X-Team-Id"] = team;
  return h;
}

export async function apiRequest(url, options = {}) {
  const { headers: optHeaders, ...rest } = options;
  const response = await fetch(url, {
    ...rest,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(optHeaders || {})
    }
  });
  const ct = response.headers.get("content-type") || "";
  let data = {};
  if (ct.includes("application/json")) {
    try {
      data = await response.json();
    } catch {
      data = {};
    }
  } else {
    try {
      data = { raw: await response.text() };
    } catch {
      data = {};
    }
  }
  if (!response.ok) {
    const msg = data.detail || data.error || data.message || `HTTP ${response.status}`;
    const err = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    err.status = response.status;
    err.payload = data;
    throw err;
  }
  return data;
}
