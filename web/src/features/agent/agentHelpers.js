export const AGENT_QUICK_PROMPTS = [
  {
    id: "eval_latest",
    label: "评价训练结果",
    text: "请基于当前 run 的指标与产物，评估最近一次训练是否达标，指出最关键的瓶颈（如精度、召回、过拟合、数据问题），并给出按优先级排序的下一步优化建议。"
  },
  {
    id: "analyze_params",
    label: "分析蒸馏参数",
    text: "请分析当前蒸馏相关参数设置，结合目标任务判断哪些参数最影响效果与稳定性，并给出可执行的调参方案（包含建议值、原因、预期收益与潜在风险）。"
  },
  {
    id: "list_runs",
    label: "实验目录",
    text: "请梳理当前实验输出目录结构，说明各关键文件/子目录的用途（如权重、日志、曲线、配置快照），并告诉我如何最快定位某次实验的核心结果。"
  },
  {
    id: "metrics_report",
    label: "指标摘要",
    text: "请生成本次训练的指标摘要：先给关键结论，再列出主要指标表现与变化趋势，最后给出 2-3 条最值得执行的后续动作。"
  }
];

const TERMINAL_TOOL_JSON_MAX = 14000;
const BLOCKED_COMMAND_HINT_RE = /执行命令\s*[（(]\s*需审批\s*[）)]/i;
const BLOCKED_DISTILL_CMD_RE = /^\s*`{0,3}\s*python\s+distill\.py\s+--config\s+configs[\\/]+distill_config\.yaml\s*`{0,3}\s*$/i;

function isBlockedLine(line) {
  const raw = typeof line === "string" ? line : "";
  const t = raw.trim();
  if (!t) return false;
  if (BLOCKED_COMMAND_HINT_RE.test(t)) return true;
  return BLOCKED_DISTILL_CMD_RE.test(t);
}

export function sanitizeBlockedCommandHints(text) {
  const raw = typeof text === "string" ? text : "";
  if (!raw) return "";
  const cleaned = raw
    .split("\n")
    .filter((line) => !isBlockedLine(line))
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return cleaned;
}

/** 从模型回复中提取「可执行」代码块（排除单行 tool JSON） */
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

/** 右侧「Agent 输出」：分节展示，避免与对话重复堆叠 */
export function formatAgentTerminalOutput(reply, toolLogs) {
  const sanitizedReply = sanitizeBlockedCommandHints(reply);
  const fences = extractExecutableFences(sanitizedReply);
  const heuristicLines = sanitizedReply
    .split("\n")
    .filter((line) => {
      const t = line.trim();
      if (!t || t.startsWith("```")) return false;
      if (isBlockedLine(t)) return false;
      return /^(python|pip|cd|git|curl|notepad|code|\$|\.\/|Invoke-)/i.test(t);
    })
    .join("\n")
    .trim();

  const sections = [];

  if (fences.length) {
    sections.push("【可执行命令 / 脚本】", fences.join("\n\n────────\n\n"));
  } else if (heuristicLines) {
    sections.push("【疑似命令行（从正文提取）】", heuristicLines);
  }

  if (toolLogs && toolLogs.length) {
    sections.push("");
    sections.push("【本地工具调用】");
    toolLogs.forEach((t, i) => {
      const name = t.call?.tool || "?";
      const args = t.call?.args && typeof t.call.args === "object" ? t.call.args : {};
      sections.push(`── ${i + 1}. ${name} ──`);
      sections.push(`请求参数:\n${JSON.stringify({ tool: name, args }, null, 2)}`);
      let resStr = "";
      try {
        resStr = JSON.stringify(t.result, null, 2);
      } catch {
        resStr = String(t.result);
      }
      const fullLen = resStr.length;
      if (fullLen > TERMINAL_TOOL_JSON_MAX) {
        resStr = `${resStr.slice(0, TERMINAL_TOOL_JSON_MAX)}\n… (以下省略 ${fullLen - TERMINAL_TOOL_JSON_MAX} 字符)`;
      }
      sections.push(`返回:\n${resStr}`);
      sections.push("");
    });
  }

  if (!sections.length) {
    return "【提示】本轮未识别到可单独摘出的终端代码块，且未经过本地工具。\n完整说明见上方对话；需要命令时请让模型用 ```powershell``` 或 ```bash``` 代码块输出。";
  }

  if (!fences.length && !heuristicLines && toolLogs?.length) {
    sections.unshift("【提示】模型未使用 Markdown 代码块给出 shell；下方为工具原始返回。");
  }

  return sections.join("\n").trim();
}

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
    /<redacted_thinking>([\s\S]*?)<\/redacted_thinking>/gi
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

export function normalizeToolName(name) {
  if (typeof name !== "string") return "";
  const raw = name.trim();
  if (!raw) return "";
  const lowered = raw.toLowerCase().replace(/[\s-]+/g, "_");
  const compact = lowered.replace(/[._]/g, "");
  const aliasMap = new Map([
    ["agent.get_context", "agent.get_context"],
    ["agent.analyze_params", "agent.analyze_params"],
    ["agent.propose_patch", "agent.propose_patch"],
    ["agent.validate_patch", "agent.validate_patch"],
    ["agent.preview_patch", "agent.preview_patch"],
    ["agent.apply_patch_with_approval", "agent.apply_patch_with_approval"],
    ["agent.list_run_history", "agent.list_run_history"],
    ["agent.rollback_run_config", "agent.rollback_run_config"],
    ["get_context", "agent.get_context"],
    ["analyze_params", "agent.analyze_params"],
    ["propose_patch", "agent.propose_patch"],
    ["validate_patch", "agent.validate_patch"],
    ["preview_patch", "agent.preview_patch"],
    ["previewpatch", "agent.preview_patch"],
    ["apply_patch_with_approval", "agent.apply_patch_with_approval"],
    ["list_run_history", "agent.list_run_history"],
    ["rollback_run_config", "agent.rollback_run_config"],
    ["agentpreviewpatch", "agent.preview_patch"]
  ]);
  if (aliasMap.has(lowered)) return aliasMap.get(lowered);
  if (aliasMap.has(compact)) return aliasMap.get(compact);
  return raw;
}

export function extractToolCallFromText(text) {
  const raw = typeof text === "string" ? text.trim() : "";
  const toToolCall = (parsed) => {
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    if (typeof parsed.tool === "string") {
      const tool = normalizeToolName(parsed.tool);
      const argsCandidate = parsed.args ?? parsed.arguments;
      let args = {};
      if (argsCandidate && typeof argsCandidate === "object" && !Array.isArray(argsCandidate)) {
        args = argsCandidate;
      } else if (typeof argsCandidate === "string") {
        try {
          const parsedArgs = JSON.parse(argsCandidate);
          if (parsedArgs && typeof parsedArgs === "object" && !Array.isArray(parsedArgs)) {
            args = parsedArgs;
          }
        } catch {
          /* ignore invalid JSON string args */
        }
      }
      return { tool, args };
    }
    if (parsed.action === "tool_call" && typeof parsed.name === "string") {
      const tool = normalizeToolName(parsed.name);
      const argsCandidate = parsed.arguments ?? parsed.args;
      let args = {};
      if (argsCandidate && typeof argsCandidate === "object" && !Array.isArray(argsCandidate)) {
        args = argsCandidate;
      } else if (typeof argsCandidate === "string") {
        try {
          const parsedArgs = JSON.parse(argsCandidate);
          if (parsedArgs && typeof parsedArgs === "object" && !Array.isArray(parsedArgs)) {
            args = parsedArgs;
          }
        } catch {
          /* ignore invalid JSON string args */
        }
      }
      return { tool, args };
    }
    return null;
  };

  const candidates = [];
  if (raw) candidates.push({ source: "raw", text: raw });
  const blocks = [...raw.matchAll(/```[a-zA-Z0-9_-]*\s*([\s\S]*?)```/g)];
  for (const m of blocks) {
    const body = (m[1] || "").trim();
    if (body) candidates.push({ source: "fence", text: body });
  }

  for (const c of candidates) {
    try {
      const parsed = JSON.parse(c.text);
      const tc = toToolCall(parsed);
      if (tc) {
        return tc;
      }
    } catch {
      /* continue */
    }
  }

  // 再做一次松散扫描：从混合文本中提取第一个可解析且符合 tool schema 的 JSON 对象
  for (let i = 0; i < raw.length; i += 1) {
    if (raw[i] !== "{") continue;
    let depth = 0;
    for (let j = i; j < raw.length; j += 1) {
      const ch = raw[j];
      if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          const slice = raw.slice(i, j + 1);
          try {
            const parsed = JSON.parse(slice);
            const tc = toToolCall(parsed);
            if (tc) {
              return tc;
            }
          } catch {
            /* try next slice */
          }
          break;
        }
      }
    }
  }

  return null;
}

/**
 * 对话气泡展示用：折叠工具 JSON、去掉中继/模型回显的 [tool:…] 行，流式末尾未闭合的工具 JSON 用短提示代替。
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
          const slice = text.slice(i, j + 1);
          const tc = extractToolCallFromText(slice);
          if (tc) return { start: i, end: j + 1, tool: tc.tool };
          break;
        }
      }
    }
  }
  return null;
}

export function softenAgentBubbleText(raw, streaming) {
  let t = sanitizeBlockedCommandHints(raw);
  t = t.replace(/\r\n/g, "\n");
  t = t.replace(/(^|\n)\s*\[tool:[^\]\n]+\]\s*/g, "$1");
  t = t.replace(/(^|\n)\s*\[(assistant|agent)\][^\n]*/gi, "$1");
  t = t.replace(/\[(assistant|agent)\]\s*/gi, "");
  let guard = 0;
  while (guard < 24) {
    guard += 1;
    const r = findFirstToolJsonRange(t);
    if (!r) break;
    t = `${t.slice(0, r.start)}\n【工具】${r.tool}\n${t.slice(r.end)}`;
  }
  if (streaming) {
    const idx = t.lastIndexOf("{");
    if (idx >= 0) {
      const tail = t.slice(idx);
      if (/"tool"\s*:/.test(tail)) {
        let depth = 0;
        for (let k = 0; k < tail.length; k += 1) {
          if (tail[k] === "{") depth += 1;
          else if (tail[k] === "}") depth -= 1;
        }
        if (depth !== 0) {
          t = `${t.slice(0, idx)}\n【工具】生成中…\n`;
        }
      }
    }
  }
  return t.replace(/\n{3,}/g, "\n\n").trimEnd();
}

/** 仅含外部 Agent 连接字段（侧栏 API 配置）的 JSON，不应触发 distill_config 审批 */
function isLikelyAgentConnectionPayloadOnly(parsed) {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return false;
  const keys = Object.keys(parsed);
  if (keys.length === 0) return false;
  const connectionKeys = new Set([
    "api_url",
    "apiUrl",
    "base_url",
    "baseUrl",
    "agent_api_url",
    "endpoint",
    "endpoint_id",
    "endpointId",
    "model",
    "api_model",
    "apiModel",
    "api_key",
    "apiKey",
    "token",
    "authorization",
    "provider",
    "region"
  ]);
  const hasDistillSection =
    (typeof parsed.distillation === "object" && parsed.distillation !== null && !Array.isArray(parsed.distillation)) ||
    (typeof parsed.training === "object" && parsed.training !== null && !Array.isArray(parsed.training)) ||
    (typeof parsed.output === "object" && parsed.output !== null && !Array.isArray(parsed.output)) ||
    (typeof parsed.wandb === "object" && parsed.wandb !== null && !Array.isArray(parsed.wandb)) ||
    (typeof parsed.patch === "object" && parsed.patch !== null);
  if (hasDistillSection) return false;
  return keys.every((k) => connectionKeys.has(k));
}

function distillPatchFromParsedObject(parsed) {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  if (isLikelyAgentConnectionPayloadOnly(parsed)) return null;
  if (parsed.patch && typeof parsed.patch === "object") return parsed.patch;
  if (typeof parsed.distillation === "object" && parsed.distillation !== null) return parsed;
  if (typeof parsed.training === "object" && parsed.training !== null) return parsed;
  if (typeof parsed.output === "object" && parsed.output !== null) return parsed;
  if (typeof parsed.wandb === "object" && parsed.wandb !== null) return parsed;
  return null;
}

/** 从混有 shell 注释的代码块中扫描平衡花括号 JSON（例如 powershell 与 JSON 同块时整段无法 JSON.parse） */
function distillPatchFromLooseText(inner) {
  const str = String(inner || "");
  for (let start = 0; start < str.length; start += 1) {
    if (str[start] !== "{") continue;
    let depth = 0;
    for (let j = start; j < str.length; j += 1) {
      const c = str[j];
      if (c === "{") depth += 1;
      else if (c === "}") {
        depth -= 1;
        if (depth === 0) {
          const slice = str.slice(start, j + 1);
          try {
            const parsed = JSON.parse(slice);
            const patch = distillPatchFromParsedObject(parsed);
            if (patch) return patch;
          } catch {
            /* 尝试下一个起始 { */
          }
          break;
        }
      }
    }
  }
  return null;
}

export function extractPatchFromResult(result, text) {
  if (result && typeof result.patch === "object" && result.patch !== null) {
    return distillPatchFromParsedObject(result.patch) ? result.patch : null;
  }
  if (result && typeof result.suggested_patch === "object" && result.suggested_patch !== null) {
    return distillPatchFromParsedObject(result.suggested_patch) ? result.suggested_patch : null;
  }
  const raw = typeof text === "string" ? text.trim() : "";
  if (!raw) return null;
  const tryJsonBlock = (inner) => {
    try {
      const parsed = JSON.parse(String(inner || "").trim());
      return distillPatchFromParsedObject(parsed);
    } catch {
      return null;
    }
  };
  const blocks = [...raw.matchAll(/```[a-zA-Z0-9_-]*\s*([\s\S]*?)```/g)];
  for (const m of blocks) {
    const body = m[1] || "";
    const patch = tryJsonBlock(body) || distillPatchFromLooseText(body);
    if (patch) return patch;
  }
  return tryJsonBlock(raw) || distillPatchFromLooseText(raw);
}

/** 同一轮 toolLogs 中，若某条之后已出现 apply/rollback（失败会抛错不会入栈），则更早的预览/提议不应再驱动审批 UI */
export function hasConfigMutationAfter(toolLogs, index) {
  for (let j = index + 1; j < toolLogs.length; j += 1) {
    const c = toolLogs[j].call?.tool;
    if (c === "agent.apply_patch_with_approval" || c === "agent.rollback_run_config") return true;
  }
  return false;
}

/** 服务端 change_summary → 聊天区可读文本（仅含本次 patch 声明的叶子字段，与审批区表格一致） */
export function formatChangeSummaryForChat(summary) {
  if (!summary || typeof summary !== "object") {
    return "【配置预览】未收到服务端变更摘要，请在下方审批区确认后执行写入。";
  }
  const paths = Array.isArray(summary.paths) ? summary.paths : [];
  if (!paths.length) {
    return "【配置预览】未检测到字段级差异（与当前配置一致或等价）。请在下方审批区确认后执行写入。";
  }
  const n = summary.stats?.changed ?? paths.length;
  const lines = paths.map((p) => {
    const { path, kind, before, after } = p;
    const b = before === undefined ? "（缺失）" : JSON.stringify(before);
    const a = after === undefined ? "（缺失）" : JSON.stringify(after);
    if (kind === "added") return `- ${path}: （新增）→ ${a}`;
    if (kind === "removed") return `- ${path}: ${b} → （删除）`;
    return `- ${path}: ${b} → ${a}`;
  });
  return [
    `【配置变更摘要】服务端核对（共 ${n} 项，以下方审批区为准）：`,
    ...lines,
    "",
    "请在对话区「批准修改训练配置」中核对后点击「让 agent 执行」写入。"
  ].join("\n");
}

/** 回合轨迹中压缩工具返回，避免气泡过大 */
export function summarizeToolResultForTrace(toolName, execResult) {
  if (execResult && typeof execResult === "object" && execResult.blocked) {
    return execResult;
  }
  if (execResult == null) return null;
  if (typeof execResult !== "object") return { raw: String(execResult) };
  const status = execResult.status;
  const name = toolName || execResult.tool;
  if (name === "agent.preview_patch" && execResult.change_summary?.stats) {
    return {
      status,
      tool: name,
      change_items: execResult.change_summary.stats.changed
    };
  }
  if (name === "agent.apply_patch_with_approval" && execResult.config) {
    return { status, tool: name, wrote_config: true };
  }
  if (name === "agent.rollback_run_config") {
    return {
      status,
      tool: name,
      rolled_back_to_version: execResult.rolled_back_to_version
    };
  }
  const raw = JSON.stringify(execResult);
  const max = 1200;
  if (raw.length <= max) return { status, tool: name || "?", body: execResult };
  return { status, tool: name || "?", truncated: `${raw.slice(0, max)}…` };
}
