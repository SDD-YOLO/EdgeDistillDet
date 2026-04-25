/**
 * utils/toolHelpers.js
 *
 * 工具名标准化与工具调用解析。
 * 所有函数均为纯函数（无副作用）。
 *
 * 对外导出：
 *   normalizeToolName         — 将各种形式的工具名统一为 "agent.xxx" 规范形式
 *   extractToolCallFromText   — 从模型回复文本中提取工具调用 { tool, args }
 *   scoreToolCallConfidence   — 评估一段文本包含工具调用的置信度（0–3）
 */

// ─────────────────────────────────────────────
// 常量
// ─────────────────────────────────────────────

const CANONICAL_TOOL_NAMES = [
  "agent.get_context",
  "agent.analyze_params",
  "agent.propose_patch",
  "agent.validate_patch",
  "agent.preview_patch",
  "agent.apply_patch_with_approval",
  "agent.list_run_history",
  "agent.rollback_run_config",
];

const EXTRA_ALIAS_MAP = new Map([
  ["previewpatch",      "agent.preview_patch"],
  ["agentpreviewpatch", "agent.preview_patch"],
]);

// ─────────────────────────────────────────────
// 内部工具函数
// ─────────────────────────────────────────────

function buildAliasMap() {
  const map = new Map(EXTRA_ALIAS_MAP);
  for (const canonical of CANONICAL_TOOL_NAMES) {
    map.set(canonical, canonical);
    map.set(canonical.replace(/^agent\./, ""), canonical);
  }
  return map;
}

const ALIAS_MAP = buildAliasMap();

function parseArgsCandidate(candidate) {
  if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) return candidate;
  if (typeof candidate === "string") {
    try {
      const p = JSON.parse(candidate);
      if (p && typeof p === "object" && !Array.isArray(p)) return p;
    } catch { /* ignore */ }
  }
  return {};
}

function toToolCall(parsed) {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  if (typeof parsed.tool === "string") {
    return { tool: normalizeToolName(parsed.tool), args: parseArgsCandidate(parsed.args ?? parsed.arguments) };
  }
  if (parsed.action === "tool_call" && typeof parsed.name === "string") {
    return { tool: normalizeToolName(parsed.name), args: parseArgsCandidate(parsed.arguments ?? parsed.args) };
  }
  return null;
}

// ─────────────────────────────────────────────
// 对外导出
// ─────────────────────────────────────────────

/**
 * 将各种形式的工具名统一为 "agent.xxx" 规范形式。
 *
 * @param {unknown} name
 * @returns {string}
 */
export function normalizeToolName(name) {
  if (typeof name !== "string") return "";
  const raw = name.trim();
  if (!raw) return "";
  const lowered = raw.toLowerCase().replace(/[\s-]+/g, "_");
  const compact  = lowered.replace(/[._]/g, "");
  return ALIAS_MAP.get(lowered) ?? ALIAS_MAP.get(compact) ?? raw;
}

/**
 * 从模型回复文本中提取第一个工具调用对象 { tool, args }。
 *
 * @param {string} text
 * @returns {{ tool: string; args: object } | null}
 */
export function extractToolCallFromText(text) {
  const raw = typeof text === "string" ? text.trim() : "";
  if (!raw) return null;

  const candidates = [raw];
  for (const m of raw.matchAll(/```[a-zA-Z0-9_-]*\s*([\s\S]*?)```/g)) {
    const body = (m[1] || "").trim();
    if (body) candidates.push(body);
  }

  for (const candidate of candidates) {
    try {
      const tc = toToolCall(JSON.parse(candidate));
      if (tc) return tc;
    } catch { /* continue */ }
  }

  for (let i = 0; i < raw.length; i += 1) {
    if (raw[i] !== "{") continue;
    let depth = 0;
    for (let j = i; j < raw.length; j += 1) {
      const ch = raw[j];
      if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          try {
            const tc = toToolCall(JSON.parse(raw.slice(i, j + 1)));
            if (tc) return tc;
          } catch { /* continue */ }
          break;
        }
      }
    }
  }

  return null;
}

/**
 * 评估一段模型回复文本「包含工具调用」的置信度。
 *
 * 返回值 confidence（0–3）：
 *   3  — 整段文本就是一个合法工具调用 JSON，或恰好是单个包含工具调用的 fence block
 *   2  — 某个 fence block 内部是合法工具调用 JSON（混有自然语言）
 *   1  — 松散扫描找到工具调用 JSON（深度混在自然语言中）
 *   0  — 未找到任何工具调用
 *
 * 调用方按置信度决策：
 *   只读工具（get_context / analyze_params）：允许 confidence >= 1
 *   变更工具（propose / preview / apply）：    要求 confidence >= 2
 *
 * @param {string} text
 * @returns {{ confidence: 0|1|2|3; toolCall: { tool: string; args: object } | null }}
 */
export function scoreToolCallConfidence(text) {
  const raw = typeof text === "string" ? text.trim() : "";
  if (!raw) return { confidence: 0, toolCall: null };

  // Level 3a：整段就是合法工具调用 JSON
  if (raw.startsWith("{") && raw.endsWith("}")) {
    try {
      const tc = toToolCall(JSON.parse(raw));
      if (tc) return { confidence: 3, toolCall: tc };
    } catch { /* fall through */ }
  }

  // Level 3b：恰好是单个 fence block，内部是合法工具调用 JSON
  const singleFence = raw.match(/^```[a-zA-Z0-9_-]*\s*([\s\S]*?)```$/);
  if (singleFence) {
    try {
      const tc = toToolCall(JSON.parse((singleFence[1] || "").trim()));
      if (tc) return { confidence: 3, toolCall: tc };
    } catch { /* fall through */ }
  }

  // Level 2：某个 fence block 中包含工具调用
  for (const m of raw.matchAll(/```[a-zA-Z0-9_-]*\s*([\s\S]*?)```/g)) {
    try {
      const tc = toToolCall(JSON.parse((m[1] || "").trim()));
      if (tc) return { confidence: 2, toolCall: tc };
    } catch { /* continue */ }
  }

  // Level 1：松散扫描
  const loose = extractToolCallFromText(raw);
  if (loose) return { confidence: 1, toolCall: loose };

  return { confidence: 0, toolCall: null };
}