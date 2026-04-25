/**
 * utils/patchHelpers.js
 *
 * Patch 提取、变更摘要格式化、工具日志分析。
 * 所有函数均为纯函数（无副作用）。
 *
 * 对外导出：
 *   extractPatchFromResult      — 从工具返回 / 模型文本中提取可写入的 patch 对象
 *   hasConfigMutationAfter      — 判断同轮 toolLogs 中后续是否已有写入操作
 *   formatChangeSummaryForChat  — 将服务端 change_summary 转为聊天区可读文本
 *   summarizeToolResultForTrace — 压缩工具返回，避免轨迹气泡过大
 */

// ─────────────────────────────────────────────
// 常量：合法的「Agent 外部连接」字段白名单
// 只含这些字段的 JSON 不应触发 distill_config 审批
// ─────────────────────────────────────────────
const AGENT_CONNECTION_KEYS = new Set([
  "api_url", "apiUrl",
  "base_url", "baseUrl",
  "agent_api_url",
  "endpoint", "endpoint_id", "endpointId",
  "model", "api_model", "apiModel",
  "api_key", "apiKey",
  "token", "authorization",
  "provider", "region",
]);

// distill_config 的顶层 section 键名
const DISTILL_SECTION_KEYS = ["distillation", "training", "output", "wandb"];

// ─────────────────────────────────────────────
// 内部工具函数
// ─────────────────────────────────────────────

/**
 * 判断 parsed 是否「仅含 Agent 外部连接字段」。
 * 满足条件时不应触发 distill_config 审批流程。
 *
 * @param {unknown} parsed
 * @returns {boolean}
 */
function isAgentConnectionPayload(parsed) {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return false;
  const keys = Object.keys(parsed);
  if (keys.length === 0) return false;

  // 若含任意 distill section 或 patch 字段，直接排除
  const hasDistillSection =
    DISTILL_SECTION_KEYS.some(
      (k) =>
        typeof parsed[k] === "object" &&
        parsed[k] !== null &&
        !Array.isArray(parsed[k])
    ) || (typeof parsed.patch === "object" && parsed.patch !== null);

  if (hasDistillSection) return false;

  return keys.every((k) => AGENT_CONNECTION_KEYS.has(k));
}

/**
 * 从已解析的 JSON 对象中提取 distill patch。
 * 返回 patch 对象，或 null（不是合法 patch）。
 *
 * @param {unknown} parsed
 * @returns {object | null}
 */
function distillPatchFromObject(parsed) {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  if (isAgentConnectionPayload(parsed)) return null;

  // 明确包含 patch 字段
  if (parsed.patch && typeof parsed.patch === "object") return parsed.patch;

  // 顶层即为 distill section（直接是 config 结构）
  for (const key of DISTILL_SECTION_KEYS) {
    if (typeof parsed[key] === "object" && parsed[key] !== null) return parsed;
  }

  return null;
}

/**
 * 从混有 shell 注释的文本中逐字符扫描，
 * 提取第一个平衡花括号的合法 distill patch JSON。
 *
 * 背景：powershell 与 JSON 混写在同一代码块时，整段无法直接 JSON.parse。
 *
 * @param {string} text
 * @returns {object | null}
 */
function distillPatchFromLooseText(text) {
  const str = String(text || "");

  for (let start = 0; start < str.length; start += 1) {
    if (str[start] !== "{") continue;

    let depth = 0;
    for (let j = start; j < str.length; j += 1) {
      const ch = str[j];
      if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          const slice = str.slice(start, j + 1);
          try {
            const patch = distillPatchFromObject(JSON.parse(slice));
            if (patch) return patch;
          } catch {
            // 当前起始 { 不是合法 JSON，继续向后扫描
          }
          break;
        }
      }
    }
  }

  return null;
}

/**
 * 尝试将字符串直接 JSON.parse 后提取 patch。
 *
 * @param {string} inner
 * @returns {object | null}
 */
function tryParseJsonBlock(inner) {
  try {
    return distillPatchFromObject(JSON.parse(String(inner || "").trim()));
  } catch {
    return null;
  }
}

// ─────────────────────────────────────────────
// 对外导出
// ─────────────────────────────────────────────

/**
 * 从工具返回结果或模型回复文本中提取可写入的 distill patch 对象。
 *
 * 优先级：
 *   1. result.patch / result.suggested_patch（工具直接返回）
 *   2. 文本中的 Markdown 代码块（```json … ```）
 *   3. 文本裸 JSON 或混合文本中的松散 JSON 扫描
 *
 * @param {object | null} result  工具调用返回值
 * @param {string}        text    模型回复原文
 * @returns {object | null}
 */
export function extractPatchFromResult(result, text) {
  // 优先从结构化工具返回中取
  if (result?.patch && typeof result.patch === "object") {
    return distillPatchFromObject(result.patch) ? result.patch : null;
  }
  if (result?.suggested_patch && typeof result.suggested_patch === "object") {
    return distillPatchFromObject(result.suggested_patch) ? result.suggested_patch : null;
  }

  const raw = typeof text === "string" ? text.trim() : "";
  if (!raw) return null;

  // 扫描 Markdown 代码块
  for (const m of raw.matchAll(/```[a-zA-Z0-9_-]*\s*([\s\S]*?)```/g)) {
    const body = m[1] || "";
    const patch = tryParseJsonBlock(body) ?? distillPatchFromLooseText(body);
    if (patch) return patch;
  }

  // 回退：裸文本
  return tryParseJsonBlock(raw) ?? distillPatchFromLooseText(raw);
}

/**
 * 判断同一轮 toolLogs 中，index 之后是否已出现写入操作
 * （apply_patch_with_approval 或 rollback_run_config）。
 *
 * 用于避免更早的 preview/propose 结果重复驱动审批 UI。
 * 注：失败的写入操作会抛错，不会入栈，因此入栈即表示成功。
 *
 * @param {Array}  toolLogs
 * @param {number} index     当前条目的下标
 * @returns {boolean}
 */
export function hasConfigMutationAfter(toolLogs, index) {
  for (let j = index + 1; j < toolLogs.length; j += 1) {
    const tool = toolLogs[j].call?.tool;
    if (
      tool === "agent.apply_patch_with_approval" ||
      tool === "agent.rollback_run_config"
    ) {
      return true;
    }
  }
  return false;
}

/**
 * 将服务端 change_summary 转为聊天区可读文本。
 * 仅含本次 patch 声明的叶子字段，与审批区表格保持一致。
 *
 * @param {object | null} summary  服务端返回的 change_summary
 * @returns {string}
 */
export function formatChangeSummaryForChat(summary) {
  if (!summary || typeof summary !== "object") {
    return "【配置预览】未收到服务端变更摘要，请在下方审批区确认后执行写入。";
  }

  const paths = Array.isArray(summary.paths) ? summary.paths : [];
  if (!paths.length) {
    return "【配置预览】未检测到字段级差异（与当前配置一致或等价）。请在下方审批区确认后执行写入。";
  }

  const total = summary.stats?.changed ?? paths.length;

  const lines = paths.map(({ path, kind, before, after }) => {
    const b = before === undefined ? "（缺失）" : JSON.stringify(before);
    const a = after  === undefined ? "（缺失）" : JSON.stringify(after);
    if (kind === "added")   return `- ${path}: （新增）→ ${a}`;
    if (kind === "removed") return `- ${path}: ${b} → （删除）`;
    return `- ${path}: ${b} → ${a}`;
  });

  return [
    `【配置变更摘要】服务端核对（共 ${total} 项，以下方审批区为准）：`,
    ...lines,
    "",
    "请在对话区「批准修改训练配置」中核对后点击「让 agent 执行」写入。",
  ].join("\n");
}

/**
 * 压缩工具返回值，用于回合轨迹气泡展示，避免内容过大。
 *
 * 针对已知工具（preview_patch / apply_patch / rollback）只保留关键字段；
 * 其余工具超过 1200 字符时截断。
 *
 * @param {string}        toolName
 * @param {unknown}       execResult
 * @returns {object | null}
 */
export function summarizeToolResultForTrace(toolName, execResult) {
  // 被阻断的调用原样透传，保留 blocked 标记
  if (execResult && typeof execResult === "object" && execResult.blocked) {
    return execResult;
  }

  if (execResult == null) return null;
  if (typeof execResult !== "object") return { raw: String(execResult) };

  const { status } = execResult;
  const name = toolName || execResult.tool;

  if (name === "agent.preview_patch" && execResult.change_summary?.stats) {
    return { status, tool: name, change_items: execResult.change_summary.stats.changed };
  }

  if (name === "agent.apply_patch_with_approval" && execResult.config) {
    return { status, tool: name, wrote_config: true };
  }

  if (name === "agent.rollback_run_config") {
    return { status, tool: name, rolled_back_to_version: execResult.rolled_back_to_version };
  }

  // 通用截断
  const MAX_CHARS = 1200;
  const raw = JSON.stringify(execResult);
  if (raw.length <= MAX_CHARS) return { status, tool: name ?? "?", body: execResult };
  return { status, tool: name ?? "?", truncated: `${raw.slice(0, MAX_CHARS)}…` };
}