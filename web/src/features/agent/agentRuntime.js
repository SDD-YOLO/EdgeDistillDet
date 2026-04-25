import { invokeAgentStreamViaRelay } from "../../api/agentStreamApi";

export function parseResponsePayload(text) {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export function buildAuthHeaderCandidates(apiKey) {
  const token = String(apiKey || "").trim();
  if (!token) return [{}];
  const withBearer = /^bearer\s+/i.test(token) ? token : `Bearer ${token}`;
  const variants = [
    { Authorization: token },
    { Authorization: withBearer },
    { "x-api-key": token },
    { Authorization: withBearer, "x-api-key": token }
  ];
  const seen = new Set();
  return variants.filter((item) => {
    const key = JSON.stringify(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function buildAgentTargets(apiUrl) {
  const raw = String(apiUrl || "").trim().replace(/\/+$/, "");
  const isArkBase = /ark\./i.test(raw) && /\/api\/v\d+$/i.test(raw);
  if (isArkBase) {
    // 方舟地址走本地中转，避免浏览器直连导致跨域/证书问题。
    return [];
  }
  const openaiPath = "/v1/chat/completions";
  const targets = [];
  const hasOpenAiEndpoint = /\/v1\/chat\/completions$/i.test(raw);
  if (hasOpenAiEndpoint) {
    targets.push({ kind: "openai", url: raw });
    return targets;
  }
  targets.push({ kind: "custom", url: raw });
  if (/\/v1$/i.test(raw)) {
    targets.push({ kind: "openai", url: `${raw}/chat/completions` });
  } else {
    targets.push({ kind: "openai", url: `${raw}${openaiPath}` });
  }
  return targets;
}

export function inferRelayEndpointCandidates(baseUrl) {
  const base = String(baseUrl || "").trim().replace(/\/+$/, "");
  if (!base) return [null];
  if (/\/chat\/completions$/i.test(base) || /\/responses$/i.test(base) || /\/messages$/i.test(base)) {
    return [base, null];
  }
  // 方舟 OpenAI 兼容地址通常是 /api/v3，正确子路径是 /chat/completions，不应拼 /v1/chat/completions。
  if (/ark\./i.test(base) && /\/api\/v\d+$/i.test(base)) {
    return [`${base}/chat/completions`];
  }
  const list = [];
  if (/\/api\/v\d+$/i.test(base)) {
    list.push(`${base}/chat/completions`);
  }
  list.push(`${base}/v1/chat/completions`);
  list.push(`${base}/chat/completions`);
  list.push(null);
  return Array.from(new Set(list));
}

/** 通过本地 HTTPS 中继读取 SSE：正文与思考分流式增量 */
export async function readAgentInvokeSseStream(response, onDelta) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let reply = "";
  let reasoning = "";
  const events = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let lineEnd;
    while ((lineEnd = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, lineEnd);
      buffer = buffer.slice(lineEnd + 1);
      const trimmed = line.replace(/\r$/, "").trim();
      if (!trimmed.startsWith("data:")) continue;
      const dataStr = trimmed.slice(5).trim();
      if (!dataStr) continue;
      let ev;
      try {
        ev = JSON.parse(dataStr);
      } catch {
        continue;
      }
      if (ev.t === "content" && ev.d) {
        reply += ev.d;
        onDelta?.({ reply, reasoning, event: ev });
      } else if (ev.t === "reasoning" && ev.d) {
        reasoning += ev.d;
        onDelta?.({ reply, reasoning, event: ev });
      } else if (ev.t === "done") {
        if (typeof ev.reply === "string") reply = ev.reply;
        if (typeof ev.reasoning === "string") reasoning = ev.reasoning;
        if (Array.isArray(ev.events)) {
          events.splice(0, events.length, ...ev.events);
        }
        onDelta?.({ reply, reasoning, event: ev });
      } else if (ev.t === "error") {
        throw new Error(ev.message || "流式调用失败");
      } else if (ev.event_type) {
        events.push(ev);
        if (ev.event_type === "model_output") {
          const payload = ev.payload && typeof ev.payload === "object" ? ev.payload : {};
          if (typeof payload.reply === "string") reply = payload.reply;
          if (typeof payload.reasoning === "string") reasoning = payload.reasoning;
        } else if (ev.event_type === "done") {
          const payload = ev.payload && typeof ev.payload === "object" ? ev.payload : {};
          if (typeof payload.reply === "string") reply = payload.reply;
          if (typeof payload.reasoning === "string") reasoning = payload.reasoning;
        } else if (ev.event_type === "error") {
          const payload = ev.payload && typeof ev.payload === "object" ? ev.payload : {};
          throw new Error(payload.message || "流式调用失败");
        }
        onDelta?.({ reply, reasoning, event: ev });
      }
    }
  }
  if (buffer.trim()) {
    const trimmed = buffer.replace(/\r$/, "").trim();
    if (trimmed.startsWith("data:")) {
      const dataStr = trimmed.slice(5).trim();
      try {
        const ev = JSON.parse(dataStr);
        if (ev.t === "done") {
          if (typeof ev.reply === "string") reply = ev.reply;
          if (typeof ev.reasoning === "string") reasoning = ev.reasoning;
          if (Array.isArray(ev.events)) {
            events.splice(0, events.length, ...ev.events);
          }
          onDelta?.({ reply, reasoning, event: ev });
        }
      } catch {
        /* ignore */
      }
    }
  }
  return { reply, reasoning, events };
}

export async function streamInvokeViaRelay({
  apiUrl,
  apiKey,
  modelName,
  apiModel,
  text,
  mode,
  systemPrompt,
  onDelta,
  runId = "default",
  sessionId = "default",
  ragOptions = {},
  toolPolicy = {},
  maxSteps = 4
}) {
  const base = String(apiUrl || "").trim();
  if (/ark\./i.test(base) && /\/api\/v/i.test(base) && !String(apiModel || "").trim()) {
    throw new Error("检测到方舟地址，请先填写“模型名 / Endpoint ID”（如 ep-xxxxxx）");
  }
  const endpointCandidates = inferRelayEndpointCandidates(base);
  const errors = [];
  for (const endpoint of endpointCandidates) {
    try {
      const res = await invokeAgentStreamViaRelay({
        provider: "openai_compatible",
        api_url: base,
        api_key: String(apiKey || "").trim() || null,
        endpoint,
        model: modelName,
        temperature: 0.2,
        max_tokens: mode === "test" ? 8 : null,
        system_prompt: systemPrompt || null,
        messages: [{ role: "user", content: mode === "test" ? "ping" : text }]
        ,
        run_id: runId,
        session_id: sessionId,
        rag_options: ragOptions,
        tool_policy: toolPolicy,
        max_steps: maxSteps
      });
      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try {
          const errBody = await res.json();
          msg = errBody.error || errBody.message || msg;
        } catch {
          try {
            msg = (await res.text()) || msg;
          } catch {
            /* ignore */
          }
        }
        throw new Error(msg);
      }
      const { reply, reasoning, events } = await readAgentInvokeSseStream(res, onDelta);
      const payload = {
        status: "ok",
        reply,
        reasoning,  // Keep empty string if model doesn't return reasoning
        events
      };
      return { payload, target: { kind: "backend-relay", url: endpoint || base } };
    } catch (error) {
      let msg = error.message;
      if (!String(apiModel || "").trim() && /NotFound|does not exist|InvalidEndpointOrModel/i.test(String(msg || ""))) {
        msg = `${msg}（请在“模型名 / Endpoint ID”填写可用模型，如方舟控制台中的 endpoint-id）`;
      }
      errors.push(`relay@${endpoint || base}: ${msg}`);
    }
  }
  throw new Error(errors.slice(0, 3).join(" | ") || "本地中转流式调用失败");
}

export async function requestAgentWithFallback({
  apiUrl,
  apiKey,
  apiModel,
  modelName,
  text,
  mode,
  systemPrompt,
  onDelta,
  onRelayFallback,
  runId = "default",
  sessionId = "default",
  ragOptions = {},
  toolPolicy = {},
  maxSteps = 4
}) {
  const trimmedApiUrl = String(apiUrl || "").trim();
  if (/ark\./i.test(trimmedApiUrl) && /\/api\/v/i.test(trimmedApiUrl) && !String(apiModel || "").trim()) {
    throw new Error("检测到方舟地址，请先填写“模型名 / Endpoint ID”（如 ep-xxxxxx）");
  }
  const targets = buildAgentTargets(trimmedApiUrl);
  const authHeaders = buildAuthHeaderCandidates(apiKey);
  const errors = [];
  const prefersRelay = /^https?:\/\//i.test(trimmedApiUrl);
  const isArkBase = /ark\./i.test(trimmedApiUrl) && /\/api\/v\d+$/i.test(trimmedApiUrl);
  if (prefersRelay) {
    try {
      return await streamInvokeViaRelay({
        apiUrl,
        apiKey,
        modelName,
        apiModel,
        text,
        mode,
        systemPrompt,
        onDelta,
        runId,
        sessionId,
        ragOptions,
        toolPolicy,
        maxSteps
      });
    } catch (error) {
      errors.push(error.message);
      onRelayFallback?.();
    }
  }
  if (isArkBase) {
    throw new Error(`${errors.slice(0, 2).join(" | ") || "本地中转调用失败"} | 请确认网络可访问方舟域名，且“模型名 / Endpoint ID”填写正确（如 ep-xxxxxx）`);
  }
  for (const target of targets) {
    for (const auth of authHeaders) {
      let body;
      if (target.kind === "openai") {
        const baseMessages = mode === "test" ? [{ role: "user", content: "ping" }] : [{ role: "user", content: text }];
        const withSystem = systemPrompt ? [{ role: "system", content: systemPrompt }, ...baseMessages] : baseMessages;
        body = mode === "test"
          ? { model: modelName, messages: withSystem, max_tokens: 1 }
          : { model: modelName, messages: withSystem, temperature: 0.2 };
      } else {
        body = mode === "test" ? { action: "ping", params: {} } : { query: text };
      }
      try {
        const res = await fetch(target.url, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...auth },
          body: JSON.stringify(body)
        });
        const payload = parseResponsePayload(await res.text());
        if (!res.ok) {
          const message = typeof payload === "object" && payload ? payload.error || payload.message : String(payload || "");
          throw new Error(message || `HTTP ${res.status}`);
        }
        return { payload, target };
      } catch (error) {
        let msg = error.message;
        if (!String(apiModel || "").trim() && /NotFound|does not exist|InvalidEndpointOrModel/i.test(String(msg || ""))) {
          msg = `${msg}（请在“模型名 / Endpoint ID”填写可用模型）`;
        }
        errors.push(`${target.kind}@${target.url}: ${msg}`);
      }
    }
  }
  throw new Error(errors.slice(0, 3).join(" | ") || "所有连接方式均失败");
}
