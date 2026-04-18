export const AGENT_QUICK_PROMPTS = [
  {
    id: "eval_latest",
    label: "评价训练结果",
    text: "请结合当前训练指标与输出目录，评价最近一次训练结果，并指出主要问题与改进方向；如需改配置，请用 Markdown 代码块给出可在 PowerShell 中执行的命令。"
  },
  {
    id: "analyze_params",
    label: "分析蒸馏参数",
    text: "请先按约定输出 tool JSON 调用 agent.get_context，再调用 agent.analyze_params 获取事实，最后给出优化建议；涉及修改 configs/distill_config.yaml 时，请用 Markdown 代码块输出 PowerShell 命令（如 notepad、code、或 Python 一行读写），不要只给泛泛而谈。"
  },
  {
    id: "list_runs",
    label: "实验目录",
    text: "请使用工具说明当前训练输出目录、exp 命名规则，以及如何定位某次实验的权重与曲线。"
  },
  {
    id: "metrics_report",
    label: "指标摘要",
    text: "请根据当前训练指标生成简短摘要，并给出下一步建议；若需要本地命令，请用代码块给出。"
  }
];

const TERMINAL_TOOL_JSON_MAX = 14000;

/** 从模型回复中提取「可执行」代码块（排除单行 tool JSON） */
export function extractExecutableFences(reply) {
  const raw = typeof reply === "string" ? reply : "";
  const out = [];
  const fenceRe = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let m;
  while ((m = fenceRe.exec(raw)) !== null) {
    const inner = (m[2] || "").trim();
    if (!inner) continue;
    if (/^\s*\{\s*"tool"\s*:/.test(inner) && inner.length < 1200) continue;
    const lang = (m[1] || "").trim().toLowerCase();
    if (lang === "json" && /"tool"\s*:/.test(inner)) continue;
    out.push(inner);
  }
  return out;
}

/** 右侧「Agent 输出」：分节展示，避免与对话重复堆叠 */
export function formatAgentTerminalOutput(reply, toolLogs) {
  const fences = extractExecutableFences(reply);
  const heuristicLines = (typeof reply === "string" ? reply : "")
    .split("\n")
    .filter((line) => {
      const t = line.trim();
      if (!t || t.startsWith("```")) return false;
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
    return "【提示】本轮未识别到可单独摘出的终端代码块，且未经过本地工具。\n完整说明见左侧对话；需要命令时请让模型用 ```powershell``` 或 ```bash``` 代码块输出。";
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
  const apiReason =
    typeof reasoningFromApi === "string" && reasoningFromApi.trim() ? reasoningFromApi.trim() : "";
  return { displayReply: main, displayReasoning: apiReason };
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
