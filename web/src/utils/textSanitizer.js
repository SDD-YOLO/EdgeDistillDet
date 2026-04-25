/**
 * utils/textSanitizer.js
 *
 * 模型回复文本的过滤、清洁与格式化。
 * 所有函数均为纯函数（无副作用）。
 *
 * 对外导出：
 *   sanitizeBlockedCommandHints  — 过滤被阻断命令提示行
 *   extractExecutableFences      — 提取可执行代码块
 *   formatAgentTerminalOutput    — 格式化右侧「Agent 终端输出」面板
 *   softenAgentBubbleText        — 净化对话气泡文本（折叠工具 JSON、处理流式尾部）
 */

import { extractToolCallFromText } from "./toolHelpers";

// ─────────────────────────────────────────────
// 常量
// ─────────────────────────────────────────────

/** 工具 JSON 在终端面板中的最大展示字符数 */
const TERMINAL_TOOL_JSON_MAX = 14_000;

/** 匹配「执行命令（需审批）」类提示文本 */
const BLOCKED_COMMAND_HINT_RE = /执行命令\s*[（(]\s*需审批\s*[）)]/i;

/** 匹配直接调用 distill.py 的命令行（带或不带反引号） */
const BLOCKED_DISTILL_CMD_RE =
  /^\s*`{0,3}\s*python\s+distill\.py\s+--config\s+configs[\\/]+distill_config\.yaml\s*`{0,3}\s*$/i;

/** 启发式命令行前缀（用于从正文中识别疑似 shell 命令） */
const HEURISTIC_CMD_PREFIX_RE =
  /^(python|pip|cd|git|curl|notepad|code|\$|\.\/|Invoke-)/i;

// ─────────────────────────────────────────────
// 内部工具函数
// ─────────────────────────────────────────────

/**
 * 判断单行文本是否应被过滤（被阻断的命令提示）。
 *
 * @param {string} line
 * @returns {boolean}
 */
function isBlockedLine(line) {
  const t = (typeof line === "string" ? line : "").trim();
  if (!t) return false;
  return BLOCKED_COMMAND_HINT_RE.test(t) || BLOCKED_DISTILL_CMD_RE.test(t);
}

/**
 * 从文本中找到第一个完整工具调用 JSON 的字符范围。
 * 用于 softenAgentBubbleText 的迭代替换。
 *
 * @param {string} raw
 * @returns {{ start: number; end: number; tool: string } | null}
 */
function findFirstToolJsonRange(raw) {
  const text = typeof raw === "string" ? raw : "";

  for (let i = 0; i < text.length; i += 1) {
    if (text[i] !== "{") continue;

    let depth = 0;
    for (let j = i; j < text.length; j += 1) {
      const ch = text[j];
      if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          const tc = extractToolCallFromText(text.slice(i, j + 1));
          if (tc) return { start: i, end: j + 1, tool: tc.tool };
          break;
        }
      }
    }
  }

  return null;
}

// ─────────────────────────────────────────────
// 对外导出
// ─────────────────────────────────────────────

/**
 * 过滤文本中所有「被阻断命令提示」行，并收缩多余空行。
 *
 * @param {string} text
 * @returns {string}
 */
export function sanitizeBlockedCommandHints(text) {
  const raw = typeof text === "string" ? text : "";
  if (!raw) return "";

  return raw
    .split("\n")
    .filter((line) => !isBlockedLine(line))
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/**
 * 从模型回复中提取「可执行」代码块，排除以下内容：
 * - 被阻断的命令行
 * - 单行 tool JSON（长度 < 1200 且以 `{"tool":` 开头）
 * - lang 为 json 且含 `"tool":` 的代码块
 *
 * @param {string} reply
 * @returns {string[]}
 */
export function extractExecutableFences(reply) {
  const raw = sanitizeBlockedCommandHints(reply);
  const out = [];
  const fenceRe = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let m;

  while ((m = fenceRe.exec(raw)) !== null) {
    const inner = (m[2] || "").trim();
    if (!inner) continue;
    if (isBlockedLine(inner)) continue;
    if (/^\s*\{\s*"tool"\s*:/.test(inner) && inner.length < 1200) continue;

    const lang = (m[1] || "").trim().toLowerCase();
    if (lang === "json" && /"tool"\s*:/.test(inner)) continue;

    out.push(inner);
  }

  return out;
}

/**
 * 格式化右侧「Agent 终端输出」面板内容，分节展示，避免与对话区重复堆叠。
 *
 * 输出节优先级：
 *   1. 可执行代码块（Markdown fence）
 *   2. 启发式命令行（从正文中识别）
 *   3. 本地工具调用日志
 *
 * @param {string}   reply     模型回复原文
 * @param {Array}    toolLogs  本轮工具调用日志
 * @returns {string}
 */
export function formatAgentTerminalOutput(reply, toolLogs) {
  const sanitizedReply = sanitizeBlockedCommandHints(reply);
  const fences = extractExecutableFences(sanitizedReply);

  const heuristicLines = sanitizedReply
    .split("\n")
    .filter((line) => {
      const t = line.trim();
      if (!t || t.startsWith("```")) return false;
      if (isBlockedLine(t)) return false;
      return HEURISTIC_CMD_PREFIX_RE.test(t);
    })
    .join("\n")
    .trim();

  const sections = [];

  // 节 1：可执行代码块 / 疑似命令行
  if (fences.length) {
    sections.push("【可执行命令 / 脚本】", fences.join("\n\n────────\n\n"));
  } else if (heuristicLines) {
    sections.push("【疑似命令行（从正文提取）】", heuristicLines);
  }

  // 节 2：工具调用日志
  if (toolLogs?.length) {
    sections.push("", "【本地工具调用】");

    toolLogs.forEach((entry, i) => {
      const name = entry.call?.tool || "?";
      const args =
        entry.call?.args && typeof entry.call.args === "object"
          ? entry.call.args
          : {};

      sections.push(`── ${i + 1}. ${name} ──`);
      sections.push(`请求参数:\n${JSON.stringify({ tool: name, args }, null, 2)}`);

      let resStr;
      try {
        resStr = JSON.stringify(entry.result, null, 2);
      } catch {
        resStr = String(entry.result);
      }

      if (resStr.length > TERMINAL_TOOL_JSON_MAX) {
        resStr = `${resStr.slice(0, TERMINAL_TOOL_JSON_MAX)}\n… (以下省略 ${resStr.length - TERMINAL_TOOL_JSON_MAX} 字符)`;
      }

      sections.push(`返回:\n${resStr}`, "");
    });
  }

  if (!sections.length) {
    return (
      "【提示】本轮未识别到可单独摘出的终端代码块，且未经过本地工具。\n" +
      "完整说明见上方对话；需要命令时请让模型用 ```powershell``` 或 ```bash``` 代码块输出。"
    );
  }

  if (!fences.length && !heuristicLines && toolLogs?.length) {
    sections.unshift("【提示】模型未使用 Markdown 代码块给出 shell；下方为工具原始返回。");
  }

  return sections.join("\n").trim();
}

/**
 * 净化对话气泡文本，用于聊天区展示。
 *
 * 处理内容：
 * - 过滤被阻断命令提示行
 * - 去除 [tool:…] / [assistant] / [agent] 等中继标记行
 * - 将工具调用 JSON 折叠为「【工具】tool_name」短标记
 * - 流式输出时，将末尾未闭合的工具 JSON 替换为「【工具】生成中…」
 *
 * @param {string}  raw        模型回复原文
 * @param {boolean} streaming  是否处于流式输出状态
 * @returns {string}
 */
export function softenAgentBubbleText(raw, streaming) {
  let t = sanitizeBlockedCommandHints(raw);
  t = t.replace(/\r\n/g, "\n");

  // 新增：提取 JSON 中的 reply 字段（处理多 JSON 拼接的情况）
  // 尝试找到第一个有效的 JSON 对象（含 reply 字段）
  const trimmed = t.trim();
  
  // 使用非贪婪匹配，找到第一个完整的 JSON 对象
  // 匹配 { ... }，但避免跨对象匹配
  const jsonMatches = trimmed.match(/\{[^{}]*("reply"[^{}]*)[^{}]*\}/g) || 
                      trimmed.match(/\{[\s\S]*?"reply"[\s\S]*?\}(?=\s*\{|$)/);
  
  if (jsonMatches && jsonMatches[0]) {
    try {
      const parsed = JSON.parse(jsonMatches[0]);
      if (parsed && typeof parsed.reply === "string") {
        t = parsed.reply;
      }
    } catch (e) {
      // 尝试处理嵌套引号问题：把字符串中的 \n 先还原
      try {
        const cleaned = jsonMatches[0]
          .replace(/\\n/g, '\n')
          .replace(/\\"/g, '"')
          .replace(/\\\\/g, '\\');
        const parsed = JSON.parse(cleaned);
        if (parsed && typeof parsed.reply === "string") {
          t = parsed.reply;
        }
      } catch (e2) {
        // 仍然失败，尝试提取 reply 字段的原始内容
        const replyMatch = jsonMatches[0].match(/"reply"\s*:\s*"([\s\S]*?)"\s*,\s*"reasoning"/);
        if (replyMatch) {
          t = replyMatch[1]
            .replace(/\\n/g, '\n')
            .replace(/\\"/g, '"')
            .replace(/\\\\/g, '\\');
        }
      }
    }
  }

  // 去除中继标记行
  t = t.replace(/(^|\n)\s*\[tool:[^\]\n]+\]\s*/g, "$1");
  t = t.replace(/(^|\n)\s*\[(assistant|agent)\][^\n]*/gi, "$1");
  t = t.replace(/\[(assistant|agent)\]\s*/gi, "");

  // 迭代折叠工具 JSON（最多 24 次防止死循环）
  for (let guard = 0; guard < 24; guard += 1) {
    const r = findFirstToolJsonRange(t);
    if (!r) break;
    t = `${t.slice(0, r.start)}\n【工具】${r.tool}\n${t.slice(r.end)}`;
  }

  // 流式尾部：未闭合的工具 JSON 替换为「生成中」占位
  if (streaming) {
    const idx = t.lastIndexOf("{");
    if (idx >= 0) {
      const tail = t.slice(idx);
      if (/"tool"\s*:/.test(tail)) {
        let depth = 0;
        for (const ch of tail) {
          if (ch === "{") depth += 1;
          else if (ch === "}") depth -= 1;
        }
        if (depth !== 0) {
          t = `${t.slice(0, idx)}\n【工具】生成中…\n`;
        }
      }
    }
  }

  return t.replace(/\n{3,}/g, "\n\n").trimEnd();
}