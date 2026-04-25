import { sanitizeBlockedCommandHints } from "./textSanitizer";

/** 从 OpenAI 风格 message.content 提取文本（多模态为数组时拼接） */
function pieceTextFromContent(val) {
  if (val == null) return "";
  if (typeof val === "string") return val;
  if (Array.isArray(val)) {
    const parts = [];
    for (const item of val) {
      if (typeof item === "string") parts.push(item);
      else if (item && typeof item === "object") {
        if (typeof item.text === "string") parts.push(item.text);
        else if (typeof item.content === "string") parts.push(item.content);
      }
    }
    return parts.join("");
  }
  return String(val);
}

/** 从 API 响应中同时取出正文与 reasoning（与后端 _extract_openai_reasoning_from_message 对齐） */
export function extractReplyAndReasoningFromPayload(payload) {
  if (typeof payload === "string") return { reply: payload, reasoning: "" };
  if (!payload || typeof payload !== "object") return { reply: String(payload ?? ""), reasoning: "" };
  let reasoning = "";
  if (typeof payload.reasoning === "string" && payload.reasoning.trim()) {
    reasoning = payload.reasoning.trim();
  }
  const msg = payload?.choices?.[0]?.message;
  if (msg && typeof msg === "object") {
    if (!reasoning) {
      const r = msg.reasoning_content ?? msg.reasoning;
      if (typeof r === "string" && r.trim()) reasoning = r.trim();
    }
    const reply = pieceTextFromContent(msg.content);
    if (reply.trim()) return { reply, reasoning };
  }
  const fallback = payload.reply ?? payload.message ?? payload.output;
  if (typeof fallback === "string") return { reply: fallback, reasoning };
  if (fallback !== undefined && fallback !== null) {
    return { reply: typeof fallback === "object" ? JSON.stringify(fallback, null, 2) : String(fallback), reasoning };
  }
  return { reply: JSON.stringify(payload, null, 2), reasoning };
}

/** 从正文中剥离常见「思考」包裹块（不进入 UI「思考过程」区，仅清洁正文） */
function splitEmbeddedReasoningFromReply(text) {
  const raw = typeof text === "string" ? text : "";
  let main = raw;
  const patterns = [
    /<think\b[^>]*>([\s\S]*?)<\/think>/gi,
    /<think\b[^>]*>([\s\S]*?)<\/redacted_thinking>/gi,
    /<think>([\s\S]*?)<\/redacted_thinking>/gi
  ];
  for (const re of patterns) {
    main = main.replace(re, () => "");
  }
  main = main.replace(/\n{3,}/g, "\n\n").trim();
  return { main };
}

/** displayReply 始终剥离内嵌块；displayReasoning 仅来自 API */
export function buildDisplayReplyAndReasoning(rawReply, reasoningFromApi) {
  const { main } = splitEmbeddedReasoningFromReply(rawReply);
  const sanitizedMain = sanitizeBlockedCommandHints(main);
  const apiReason =
    typeof reasoningFromApi === "string" && reasoningFromApi.trim() ? reasoningFromApi.trim() : "";
  return { displayReply: sanitizedMain, displayReasoning: apiReason };
}
