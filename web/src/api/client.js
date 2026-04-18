export async function apiRequest(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options
  });
  const data = await response.json();
  if (!response.ok) {
    const err = new Error(data.error || "请求失败");
    err.status = response.status;
    err.payload = data;
    throw err;
  }
  return data;
}
