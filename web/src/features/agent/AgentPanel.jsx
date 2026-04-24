import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { executeAgentTool, fetchAgentTools, previewAgentPatch } from "../../api/agentApi";
import { fetchDistillConfig } from "../../api/configApi";
import { broadcastDistillConfigUpdate } from "../../constants/distillConfigSync";
import AgentPanelView from "./AgentPanelView";
import {
  buildDisplayReplyAndReasoning,
  extractPatchFromResult,
  extractReplyAndReasoningFromPayload,
  extractToolCallFromText,
  formatChangeSummaryForChat,
  hasConfigMutationAfter,
  sanitizeBlockedCommandHints,
  summarizeToolResultForTrace
} from "./agentHelpers";
import { requestAgentWithFallback } from "./agentRuntime";

const resolveModelName = (apiModel) => {
  const value = String(apiModel || "").trim();
  return value || "gpt-4o-mini";
};

const isArkApiUrl = (apiUrl) => {
  const text = String(apiUrl || "").trim().toLowerCase();
  return text.includes("ark.") && text.includes("/api/v");
};

const AGENT_CHAT_STORAGE_KEY = "edge_distill_agent_chat_messages_v1";
let messageIdSeq = 0;
const nextMessageId = () => `msg-${Date.now()}-${messageIdSeq++}`;

const extractAllFenceBodies = (text) => {
  const raw = typeof text === "string" ? text : "";
  const out = [];
  const re = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let m;
  while ((m = re.exec(raw)) !== null) out.push({ lang: String(m[1] || "").trim().toLowerCase(), body: String(m[2] || "") });
  return out;
};
const extractJsonCodeBlocks = (text) => {
  const fences = extractAllFenceBodies(text);
  const out = [];
  for (const f of fences) {
    const body = String(f.body || "").trim();
    if (!body) continue;
    if (f.lang === "json") {
      out.push(body);
      continue;
    }
    try {
      JSON.parse(body);
      out.push(body);
    } catch {
      /* not a JSON block */
    }
  }
  const inlineToolCall = extractToolCallFromText(text);
  if (inlineToolCall?.tool) {
    const normalized = JSON.stringify(
      {
        tool: inlineToolCall.tool,
        args: inlineToolCall.args || {}
      },
      null,
      2
    );
    if (!out.includes(normalized)) out.push(normalized);
  }
  return out;
};
const extractStandaloneToolCall = (text) => {
  const raw = typeof text === "string" ? text.trim() : "";
  if (!raw) return null;
  // strict: whole message is a single JSON object
  const direct = extractToolCallFromText(raw);
  if (direct && raw.startsWith("{") && raw.endsWith("}")) {
    return direct;
  }
  // strict: whole message is exactly one fenced block containing JSON
  const singleFence = raw.match(/^```[a-zA-Z0-9_-]*\s*([\s\S]*?)```$/);
  if (singleFence) {
    const body = String(singleFence[1] || "").trim();
    const fenced = extractToolCallFromText(body);
    if (fenced) return fenced;
  }
  return null;
};

const normalizeToolExecutionResult = (result) => {
  if (!result || typeof result !== "object") return result;
  if (
    result.status === "ok" &&
    typeof result.tool === "string" &&
    result.result &&
    typeof result.result === "object" &&
    !Array.isArray(result.result)
  ) {
    return result.result;
  }
  return result;
};

const isMutationIntentRequested = (text) => {
  const raw = String(text || "");
  // 明确否定改参意图时，不进入审批/写入链路
  if (
    /(不改参数|不要改参数|无需修改参数|不用修改参数|暂不修改参数|先不修改参数|仅咨询|只咨询|不要执行|不要应用|不需要执行)/i.test(
      raw
    )
  ) {
    return false;
  }
  return /(?:请|帮我|现在|直接)\s*(?:修改|调整|改)\s*(?:参数|配置)|(?:把|将)[^。\n]{0,48}(?:改为|调到|设为|更新为)|(?:修改|调整|改|更新)[^。\n]{0,48}(?:temperature|w_kd|w_feat|alpha|beta|loss|lr|学习率|蒸馏|distillation|training|output|参数|配置)|(?:生成|给我|出一版|再来一版)\s*(?:patch|补丁|参数修改方案|调参方案)|(?:预览|preview)\s*(?:patch|补丁)|(?:应用|apply|写入|执行)\s*(?:patch|补丁|参数变更)|明确修改(?:哪些)?参数|可执行的?调参方案|参数优化方案/i.test(
    raw
  );
};

const isContinuationLikeMutationText = (text) => {
  const raw = String(text || "").trim();
  if (!raw) return false;
  return /^(继续|接着|下一步|按上面|照上面|就按这个|按这个|执行|应用|确认执行|继续优化|继续调参|继续改|继续调整)/i.test(
    raw
  );
};

const shouldAutoBootstrapContext = (text) => {
  const raw = String(text || "").trim();
  if (!raw) return false;
  return /(评估|分析|指标|run|实验|目录|训练|蒸馏|参数|配置|结果|瓶颈|优化)/i.test(raw);
};

function loadStoredAgentMessages() {
  try {
    const raw = window.localStorage.getItem(AGENT_CHAT_STORAGE_KEY);
    if (!raw) return [];
    const p = JSON.parse(raw);
    if (!Array.isArray(p)) return [];
    const sanitizeTraceRounds = (rounds) => {
      if (!Array.isArray(rounds)) return rounds;
      return rounds.map((r) => {
        if (!r || typeof r !== "object") return r;
        const reply = typeof r.reply === "string" ? sanitizeBlockedCommandHints(r.reply) : r.reply;
        const reasoning = typeof r.reasoning === "string" ? sanitizeBlockedCommandHints(r.reasoning) : r.reasoning;
        const jsonCodeBlocks = Array.isArray(r.jsonCodeBlocks)
          ? r.jsonCodeBlocks.map((b) => (typeof b === "string" ? sanitizeBlockedCommandHints(b) : b))
          : r.jsonCodeBlocks;
        return { ...r, reply, reasoning, jsonCodeBlocks };
      });
    };
    const loaded = p.map((msg) => {
      if (!msg || typeof msg !== "object") return msg;
      if (msg.role !== "agent") return { ...msg, _messageId: msg._messageId || nextMessageId() };
      const content = typeof msg.content === "string" ? sanitizeBlockedCommandHints(msg.content) : msg.content;
      const reasoningApi =
        typeof msg.reasoningApi === "string" ? sanitizeBlockedCommandHints(msg.reasoningApi) : msg.reasoningApi;
      const traceRounds = sanitizeTraceRounds(msg.traceRounds);
      // 历史消息恢复时统一清掉 streaming，避免旧会话残留状态干扰本轮加载占位与流式判定
      return { ...msg, content, reasoningApi, traceRounds, streaming: false, _messageId: msg._messageId || nextMessageId() };
    });
    const repaired = [];
    for (const msg of loaded) {
      const prev = repaired[repaired.length - 1];
      if (msg?.role === "user" && prev?.role === "user") {
        repaired.push({
          role: "agent",
          kind: "history_gap",
          content: "（已修复）检测到历史会话中丢失了一条助手消息，这里为占位提示。",
          _messageId: nextMessageId()
        });
      }
      repaired.push(msg);
    }
    const loadedToolRounds = loaded.reduce((acc, m) => {
      if (m?.role !== "agent" || !Array.isArray(m?.traceRounds)) return acc;
      return (
        acc +
        m.traceRounds.filter((r) => !!r?.tool?.name || (Array.isArray(r?.jsonCodeBlocks) && r.jsonCodeBlocks.length > 0)).length
      );
    }, 0);
    const loadedStreamingCount = repaired.reduce((acc, m) => acc + (m?.role === "agent" && m?.streaming ? 1 : 0), 0);
    const loadedTailRoles = repaired.slice(-4).map((m) => String(m?.role || ""));
    return repaired;
  } catch {
    return [];
  }
}

function AgentPanel({ toast, active }) {
  const [apiUrl, setApiUrl] = useState(() => window.localStorage.getItem("edge_distill_agent_api_url") || "");
  const [apiKey, setApiKey] = useState(() => window.localStorage.getItem("edge_distill_agent_api_key") || "");
  const [apiModel, setApiModel] = useState(() => window.localStorage.getItem("edge_distill_agent_api_model") || "");
  const [messages, setMessages] = useState(() => loadStoredAgentMessages());
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [approvalOpen, setApprovalOpen] = useState(false);
  const [approvalChangeSummary, setApprovalChangeSummary] = useState(null);
  const [approvalToken, setApprovalToken] = useState("");
  const [approvalRunId, setApprovalRunId] = useState("default");
  const [approvalRequestHash, setApprovalRequestHash] = useState("");
  const chatTextareaRef = useRef(null);
  const chatMessagesRef = useRef(null);
  const lastUserMutationIntentRef = useRef(false);
  const agentSlotIndexRef = useRef(-1);
  const chatScrollRafRef = useRef(0);
  const shouldStickToBottomRef = useRef(true);

  const isNearBottom = (el, threshold = 80) => {
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
  };

  const scheduleScrollRef = useRef(() => {});
  scheduleScrollRef.current = () => {
    const el = chatMessagesRef.current;
    if (!el || !active) return;
    if (!shouldStickToBottomRef.current) return;
    if (chatScrollRafRef.current) return;
    chatScrollRafRef.current = window.requestAnimationFrame(() => {
      chatScrollRafRef.current = 0;
      const box = chatMessagesRef.current;
      if (!box || !active) return;
      const settleToBottom = (attempt = 0) => {
        box.scrollTop = box.scrollHeight;
        shouldStickToBottomRef.current = true;
        const gap = Math.round(box.scrollHeight - box.scrollTop - box.clientHeight);
        if (gap <= 1 || attempt >= 3) return;
        window.requestAnimationFrame(() => {
          const latestBox = chatMessagesRef.current;
          if (!latestBox || !active || !shouldStickToBottomRef.current) return;
          settleToBottom(attempt + 1);
        });
      };
      settleToBottom();
      shouldStickToBottomRef.current = true;
      const gapAfterScroll = Math.round(box.scrollHeight - box.scrollTop - box.clientHeight);
      window.requestAnimationFrame(() => {
        const finalBox = chatMessagesRef.current;
        if (!finalBox || !active) return;
        const finalGap = Math.round(finalBox.scrollHeight - finalBox.scrollTop - finalBox.clientHeight);
        if (finalGap > 1 && shouldStickToBottomRef.current) {
          const recoverSettle = (attempt = 0) => {
            const currentBox = chatMessagesRef.current;
            if (!currentBox || !active || !shouldStickToBottomRef.current) return;
            currentBox.scrollTop = currentBox.scrollHeight;
            const recoverGap = Math.round(currentBox.scrollHeight - currentBox.scrollTop - currentBox.clientHeight);
            if (recoverGap <= 1 || attempt >= 2) return;
            window.requestAnimationFrame(() => recoverSettle(attempt + 1));
          };
          recoverSettle();
        }
      });
    });
  };

  useLayoutEffect(() => {
    if (!active) return;
    scheduleScrollRef.current();
  }, [messages, loading, active]);

  useEffect(() => {
    const el = chatMessagesRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const onScroll = () => {
      shouldStickToBottomRef.current = isNearBottom(el);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    const ro = new ResizeObserver(() => {
      if (!active) return;
      if (!shouldStickToBottomRef.current) return;
      scheduleScrollRef.current();
    });
    ro.observe(el);
    const mo = new MutationObserver(() => {
      if (!active) return;
      if (!shouldStickToBottomRef.current) return;
      scheduleScrollRef.current();
    });
    mo.observe(el, { childList: true, subtree: true, characterData: true });
    return () => {
      ro.disconnect();
      mo.disconnect();
      el.removeEventListener("scroll", onScroll);
    };
  }, [active]);

  const resizeChatInput = () => {
    const el = chatTextareaRef.current;
    if (!el) return;
    const computed = window.getComputedStyle(el);
    const lineHeight = parseFloat(computed.lineHeight) || 20;
    const paddingTop = parseFloat(computed.paddingTop) || 0;
    const paddingBottom = parseFloat(computed.paddingBottom) || 0;
    const borderTop = parseFloat(computed.borderTopWidth) || 0;
    const borderBottom = parseFloat(computed.borderBottomWidth) || 0;
    const isBorderBox = computed.boxSizing === "border-box";
    const maxHeight = lineHeight * 3 + paddingTop + paddingBottom + borderTop + borderBottom;
    el.style.height = "auto";
    const postAutoScrollHeight = el.scrollHeight;
    const requiredHeight = postAutoScrollHeight + (isBorderBox ? borderTop + borderBottom : 0);
    const appliedHeight = Math.min(requiredHeight, maxHeight);
    el.style.height = `${appliedHeight}px`;
    el.style.overflowY = requiredHeight > maxHeight ? "auto" : "hidden";
  };

  useEffect(() => {
    if (!chatTextareaRef.current) return;
    resizeChatInput();
  }, [input]);

  useEffect(() => {
    try {
      window.localStorage.setItem(AGENT_CHAT_STORAGE_KEY, JSON.stringify(messages));
    } catch {
      /* quota or private mode */
    }
  }, [messages]);

  const exportAgentSession = () => {
    try {
      const blob = new Blob(
        [JSON.stringify({ exportedAt: new Date().toISOString(), messages }, null, 2)],
        { type: "application/json" }
      );
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `edgedistilldet-agent-chat-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
      toast("已导出会话 JSON", "success");
    } catch (error) {
      toast(`导出失败: ${error.message}`, "error");
    }
  };

  const saveConfig = () => {
    window.localStorage.setItem("edge_distill_agent_api_url", apiUrl.trim());
    window.localStorage.setItem("edge_distill_agent_api_key", apiKey.trim());
    window.localStorage.setItem("edge_distill_agent_api_model", apiModel.trim());
    toast("Agent API 配置已保存", "success");
  };

  const buildAgentSystemPrompt = async () => {
    let contract = null;
    try {
      contract = await fetchAgentTools();
    } catch {
      contract = null;
    }
    const toolList = Array.isArray(contract?.tools)
      ? contract.tools.map((t) => `- ${t.name}: input=${JSON.stringify(t.input || {})}, output=${t.output || ""}`).join("\n")
      : "- agent.get_context\n- agent.analyze_params\n- agent.propose_patch\n- agent.validate_patch\n- agent.preview_patch\n- agent.apply_patch_with_approval\n- agent.list_run_history\n- agent.rollback_run_config";
    return [
      "你是训练参数优化 Agent。你可以且应该使用工具先获取事实，再给结论。",
      "可用工具如下：",
      toolList,
      "",
      "当需要调用工具时，你必须只输出一个 JSON 对象（不要输出其他文本）：",
      '{"tool":"agent.get_context","args":{"run_id":"default"}}',
      "工具结果会在下一轮以 tool 消息回传给你。",
      "**修改 distill 配置（configs/distill_config.yaml）时**：优先输出 `:{\"tool\":\"agent.preview_patch\",\"args\":{\"patch\":{...}}}`（可先 `agent.validate_patch`）；若只调用了 `agent.propose_patch`，界面也会自动用返回的 patch 请求预览并在对话区显示「批准修改训练配置」。**禁止**在最终答复里用「是否需要我执行/是否生成补丁」等话术向用户索要确认。",
      "**`patch` 必须与审批内容严格一致**：只包含你打算改动的顶层段（distillation / training / output）及其字段；**禁止**在 patch 里附带未在答复中说明的参数（例如只讨论 kd loss 却写入 temperature）。自然语言列出的每一项都必须在 patch 中出现且数值一致。",
      "在输出 tool JSON 前后，用一两句话概括拟修改的字段与意图；**字段级旧值→新值以对话区审批摘要与聊天中的服务端摘要为准**，避免与 patch 不一致。",
      "终端里的训练命令（```powershell``` / ```bash```）只能作为补充说明；真正写入配置必须经过上述工具链或界面审批。",
      "禁止输出「执行命令（需审批）」这类审批提示模板；禁止建议 `python distill.py --config configs/distill_config.yaml`（仓库无该入口）。如需训练命令，使用项目实际入口（如 `python main.py train --config configs/distill_config.yaml`）。",
      "当不再需要工具、输出最终答复时：用自然语言说明变更理由与风险；若已调用 `agent.preview_patch`，只需提示用户在对话区「批准修改训练配置」中批准，不要重复询问。仅在确定无法走工具链时，才用 JSON 代码块给出结构化 patch 作为兜底。",
      "**审批/写入流程结束后**：若用户未主动要求继续改配置，你**不得**主动提出新的修改建议、不得追问「要不要继续调整」「是否还要改某参数」「需不需要再预览一版」等；**不得**自动再发起 `agent.propose_patch` / `agent.preview_patch` 或引导用户进入下一轮审批。此时只做简短收尾（例如已写入/已按对话区审批操作即可训练），然后停止。",
      "**仅当用户明确说出**要改配置、改某字段、再优化、再出一版 patch 等意图时，你才可以再次使用配置相关工具或给出修改建议。"
    ].join("\n");
  };

  const runAgentWithTools = async ({
    userText,
    maxRounds = 6,
    forceAllowMutationTools = false,
    turnStartTs = Date.now()
  }) => {
    agentSlotIndexRef.current = -1;
    let streamedToolJsonSeen = false;
    let streamingTraceDraft = null;
    let missingSlotLogged = false;
    const systemPrompt = await buildAgentSystemPrompt();
    const convo = [...messages, { role: "user", content: userText }];
    const toolLogs = [];
    const traceRounds = [];
    const mutationTools = new Set([
      "agent.propose_patch",
      "agent.preview_patch",
      "agent.apply_patch_with_approval",
      "agent.rollback_run_config"
    ]);
    const userTextRaw = String(userText || "");
    const explicitUserToolCall = extractToolCallFromText(userTextRaw);
    const allowMutationByIntent = isMutationIntentRequested(userTextRaw) || forceAllowMutationTools;
    const allowMutationByExplicitToolCall =
      !!explicitUserToolCall && mutationTools.has(String(explicitUserToolCall.tool || ""));
    const allowMutationTools = allowMutationByIntent || allowMutationByExplicitToolCall;
    const canAutoBridgeFromAnalysis = () => {
      if (!allowMutationTools) return false;
      const hasAnalyze = toolLogs.some((t) => t.call?.tool === "agent.analyze_params");
      if (!hasAnalyze) return false;
      const hasPatchFlow = toolLogs.some((t) => mutationTools.has(t.call?.tool));
      return !hasPatchFlow;
    };
    const prefersRelayGlobal = /^https?:\/\//i.test(apiUrl.trim());
    const defaultContinue = "请继续。若需工具则按约定输出 tool JSON。";
    const readOnlyContinue =
      "请继续完成分析结论，但本轮用户未要求改参数。禁止调用 agent.propose_patch / agent.preview_patch / agent.apply_patch_with_approval / agent.rollback_run_config；若你认为值得调整参数，仅用自然语言一句话询问「是否需要我提供参数修改方案」。";
    /** 写入/回滚完成后若仍用「请继续+tool JSON」，模型会再出一轮 patch，形成审批死循环 */
    const finalizeAfterMutation =
      "配置变更已通过工具落盘。请仅用一两句自然语言确认完成（不要输出 JSON 代码块、不要调用任何工具）。在用户未明确要求继续改配置前，禁止主动提出修改建议、禁止追问是否继续调整、禁止再次调用 propose_patch / preview_patch / apply_patch / rollback 相关工具。";
    let continuationSuffix = allowMutationTools ? defaultContinue : readOnlyContinue;

    const flushTrace = (reply, reasoning, toolInfo, execRes) => {
      const replyJsonBlocks = extractJsonCodeBlocks(reply);
      const reasoningJsonBlocks = extractJsonCodeBlocks(reasoning);
      const jsonCodeBlocks = [...replyJsonBlocks, ...reasoningJsonBlocks];
      const hasRealToolRound = !!(toolInfo?.tool || jsonCodeBlocks.length > 0);
      streamingTraceDraft = null;
      if (hasRealToolRound) {
        traceRounds.push({
          round: traceRounds.length + 1,
          reply,
          reasoning: reasoning || "",
          jsonCodeBlocks,
          tool: toolInfo ? { name: toolInfo.tool, args: toolInfo.args || {} } : null,
          toolResultSummary:
            execRes !== undefined && execRes !== null
              ? summarizeToolResultForTrace(toolInfo?.tool, execRes)
              : undefined
        });
      }
      if (prefersRelayGlobal && agentSlotIndexRef.current >= 0) {
        setMessages((prev) => {
          const next = [...prev];
          const idx = agentSlotIndexRef.current;
          if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
            const hasAnyToolRound = traceRounds.some((r) => !!r.tool?.name);
            next[idx] = { ...next[idx], traceRounds: [...traceRounds], traceOpen: hasAnyToolRound };
          }
          return next;
        });
      }
    };

    for (let round = 0; round < maxRounds; round += 1) {
      const transcriptText = convo
        .map((m) => {
          if (m.role === "tool") {
            return `[tool:${m.name}]\n${m.content}`;
          }
          return `[${m.role}] ${m.content}`;
        })
        .join("\n\n");
      const prompt = `${transcriptText}\n\n${continuationSuffix}`;
      const prefersRelay = prefersRelayGlobal;
      const resolvedModelName = resolveModelName(apiModel);

      if (prefersRelay && round === 0) {
        setMessages((prev) => {
          const idx = prev.length;
          agentSlotIndexRef.current = idx;
          return [
            ...prev,
            {
              _messageId: nextMessageId(),
              role: "agent",
              content: "",
              reasoningApi: "",
              toolsUsed: [],
              streaming: true,
              traceOpen: false,
              modelName: resolvedModelName
            }
          ];
        });
      } else if (prefersRelay && round > 0) {
        setMessages((prev) => {
          const next = [...prev];
          const idx = agentSlotIndexRef.current;
          if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
            const shouldClearStaleContent = traceRounds.length > 0;
            if (shouldClearStaleContent) {
            }
            next[idx] = {
              ...next[idx],
              streaming: true,
              ...(shouldClearStaleContent ? { content: "" } : {})
            };
          }
          return next;
        });
      }

      const onDelta = prefersRelay
        ? ({ reply, reasoning }) => {
            const maybeTool = extractToolCallFromText(reply);
            const jsonBlocks = extractJsonCodeBlocks(reply);
            if (maybeTool?.tool) {
              if (!streamedToolJsonSeen) {
                streamedToolJsonSeen = true;
              }
              streamingTraceDraft = {
                round: traceRounds.length + 1,
                reply: String(reply || ""),
                reasoning: String(reasoning || ""),
                jsonCodeBlocks: jsonBlocks,
                tool: { name: maybeTool.tool, args: maybeTool.args || {} },
                toolResultSummary: { status: "streaming" }
              };
            }
            setMessages((prev) => {
              const next = [...prev];
              const idx = agentSlotIndexRef.current;
              if (idx >= 0 && idx < next.length && next[idx].role === "agent" && next[idx].streaming) {
                const traceForUi =
                  streamingTraceDraft && streamedToolJsonSeen
                    ? [...traceRounds, streamingTraceDraft]
                    : next[idx].traceRounds || [...traceRounds];
                if (round > 0 && traceRounds.length > 0 && (!Array.isArray(next[idx].traceRounds) || next[idx].traceRounds.length === 0)) {
                }
                const suppressStreamingText =
                  !maybeTool?.tool && round > 0 && Array.isArray(traceForUi) && traceForUi.length > 0;
                const contentForUi = maybeTool?.tool || suppressStreamingText ? "" : reply || "";
                if (maybeTool?.tool || suppressStreamingText) {
                }
                // Show actual model output, not placeholder text
                // Tool rounds will be shown in traceRounds
                const keepTraceOpen =
                  !!(
                    next[idx].traceOpen ||
                    (streamingTraceDraft && streamedToolJsonSeen) ||
                    traceRounds.some((r) => !!r.tool?.name)
                  );
                next[idx] = {
                  ...next[idx],
                  content: contentForUi,
                  reasoningApi: reasoning || "",
                  streaming: true,
                  traceRounds: traceForUi,
                  traceOpen: keepTraceOpen,
                  modelName: resolvedModelName
                };
              } else if (!missingSlotLogged) {
                missingSlotLogged = true;
              }
              return next;
            });
          }
        : undefined;

      const onRelayFallback = prefersRelay
        ? () => {
            setMessages((prev) => {
              const idx = agentSlotIndexRef.current;
              const next = [...prev];
              if (idx >= 0 && idx < next.length && next[idx].streaming) {
                next.splice(idx, 1);
              }
              agentSlotIndexRef.current = -1;
              return next;
            });
          }
        : undefined;

      let payload;
      let target;
      let reply;
      let displayReply;
      let displayReasoning;
      try {
        const result = await requestAgentWithFallback({
          apiUrl,
          apiKey,
          apiModel,
          modelName: resolvedModelName,
          text: prompt,
          mode: "chat",
          systemPrompt,
          onDelta,
          onRelayFallback
        });
        payload = result.payload;
        target = result.target;
        const parsed = extractReplyAndReasoningFromPayload(payload?.reply || payload);
        reply = parsed.reply;
        const built = buildDisplayReplyAndReasoning(reply, parsed.reasoning);
        displayReply = built.displayReply;
        displayReasoning = built.displayReasoning;
        // Set reasoningApi if model returns reasoning (even if empty, to distinguish from no reasoning support)
        if (typeof payload?.reasoning === "string" && prefersRelay) {
          setMessages((prev) => {
            const next = [...prev];
            const idx = agentSlotIndexRef.current;
            if (idx >= 0 && idx < next.length && next[idx].role === "agent" && next[idx].streaming) {
              next[idx] = { ...next[idx], reasoningApi: payload.reasoning };
            }
            return next;
          });
        }
      } catch (error) {
        if (prefersRelay) {
          setMessages((prev) => {
            const next = [...prev];
            const idx = agentSlotIndexRef.current;
            if (idx >= 0 && idx < next.length && next[idx].role === "agent" && next[idx].streaming) {
              next[idx] = { ...next[idx], streaming: false };
            }
            return next;
          });
        }
        throw error;
      }

      const looseToolCall = extractToolCallFromText(reply);
      const toolCall = extractStandaloneToolCall(reply);
      const effectiveToolCall = toolCall || (allowMutationTools ? looseToolCall : null);

      const willAutoBridge = !toolCall && canAutoBridgeFromAnalysis();
      const toolNames = toolLogs.map((t) => t.call.tool);

      if (prefersRelay && target?.kind === "backend-relay" && !effectiveToolCall && !willAutoBridge) {
        setMessages((prev) => {
          const next = [...prev];
          const idx = agentSlotIndexRef.current;
          if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
            // Use reasoning if model supports it (check if reasoningApi was ever set during streaming)
            const streamedReasoning = next[idx].reasoningApi;
            const hasReasoningSupport = typeof streamedReasoning === "string";
            const finalReasoning = hasReasoningSupport 
              ? (displayReasoning || streamedReasoning || "")
              : "";
            const traceForUi =
              streamingTraceDraft && streamedToolJsonSeen
                ? [...traceRounds, streamingTraceDraft]
                : next[idx].traceRounds || [...traceRounds];
            const hasAnyToolRound = traceForUi.some((r) => !!r.tool?.name);
            next[idx] = {
              ...next[idx],
              content: displayReply,
              toolsUsed: toolNames,
              traceRounds: traceForUi,
              traceOpen: hasAnyToolRound ? true : next[idx].traceOpen,
              modelName: payload?.model || resolvedModelName,
              streaming: false,
              // Always keep reasoningApi if model supports reasoning (even if content is empty)
              ...(hasReasoningSupport ? { reasoningApi: finalReasoning } : {})
            };
          }
          return next;
        });
      }
      // Note: We no longer replace content with placeholder text
      // The actual model output (including tool JSON) is preserved in content
      // Tool execution status is shown via traceRounds
      if (!effectiveToolCall) {
        if (round === 0 && toolLogs.length === 0 && shouldAutoBootstrapContext(userText)) {
          const bootstrapCall = { tool: "agent.get_context", args: { run_id: "default" } };
          const rawExecResult = await executeAgentTool({ tool: bootstrapCall.tool, args: bootstrapCall.args });
          const execResult = normalizeToolExecutionResult(rawExecResult);
          toolLogs.push({ call: bootstrapCall, result: execResult });
          flushTrace(displayReply, displayReasoning, bootstrapCall, execResult);
          convo.push({ role: "assistant", content: reply });
          convo.push({ role: "tool", name: bootstrapCall.tool, content: JSON.stringify(execResult, null, 2) });
          continue;
        }
        if (willAutoBridge) {
          const fallbackCall = {
            tool: "agent.propose_patch",
            args: {
              goal: String(userText || "").trim() || "根据分析结果生成可审批的配置 patch",
              constraints: {
                source: "auto_bridge_from_analyze",
                require_preview: true
              }
            }
          };
          const rawExecResult = await executeAgentTool({ tool: fallbackCall.tool, args: fallbackCall.args });
          const execResult = normalizeToolExecutionResult(rawExecResult);
          toolLogs.push({ call: fallbackCall, result: execResult });
          if (prefersRelay && agentSlotIndexRef.current >= 0) {
            const namesAfter = toolLogs.map((t) => t.call.tool);
            setMessages((prev) => {
              const next = [...prev];
              const idx = agentSlotIndexRef.current;
              if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
                next[idx] = { ...next[idx], toolsUsed: namesAfter };
              }
              return next;
            });
          }
          convo.push({ role: "assistant", content: reply });
          convo.push({
            role: "tool",
            name: fallbackCall.tool,
            content: JSON.stringify(execResult, null, 2)
          });
          continuationSuffix = allowMutationTools ? defaultContinue : readOnlyContinue;
          flushTrace(displayReply, displayReasoning, fallbackCall, execResult);
          continue;
        }
        flushTrace(displayReply, displayReasoning, null, null);
        if (prefersRelay && target?.kind === "backend-relay") {
          setMessages((prev) => {
            const next = [...prev];
            const idx = agentSlotIndexRef.current;
            if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
              next[idx] = { ...next[idx], traceOpen: false };
            }
            return next;
          });
        }
        return {
          payload,
          reply,
          displayReply,
          displayReasoning,
          target,
          toolLogs,
          streamedRelay: target?.kind === "backend-relay",
          traceRounds
        };
      }
      if (mutationTools.has(effectiveToolCall.tool) && !allowMutationTools) {
        flushTrace(displayReply, displayReasoning, effectiveToolCall, { blocked: true, reason: "mutation_not_allowed" });
        return {
          payload,
          reply,
          displayReply:
            "当前消息未明确要求修改参数，已跳过参数变更流程。是否需要我提供一版参数修改方案？",
          displayReasoning,
          target,
          toolLogs,
          streamedRelay: target?.kind === "backend-relay",
          traceRounds
        };
      }
      if (effectiveToolCall.tool === "agent.apply_patch_with_approval") {
        flushTrace(displayReply, displayReasoning, effectiveToolCall, { blocked: true, reason: "apply_via_sidebar" });
        return {
          payload,
          reply,
          displayReply:
            "已拦截自动写入请求。请先在对话区审批区核对 patch，再使用「让 agent 执行」完成写入。",
          displayReasoning,
          target,
          toolLogs,
          streamedRelay: target?.kind === "backend-relay",
          traceRounds
        };
      }
      if (!toolCall && effectiveToolCall) {
      }
      const rawExecResult = await executeAgentTool({ tool: effectiveToolCall.tool, args: effectiveToolCall.args || {} });
      const execResult = normalizeToolExecutionResult(rawExecResult);
      toolLogs.push({ call: effectiveToolCall, result: execResult });
      flushTrace(displayReply, displayReasoning, effectiveToolCall, execResult);
      if (effectiveToolCall.tool === "agent.apply_patch_with_approval" && execResult && execResult.status === "ok") {
        continuationSuffix = finalizeAfterMutation;
      } else if (effectiveToolCall.tool === "agent.rollback_run_config" && execResult && execResult.status === "ok") {
        continuationSuffix = finalizeAfterMutation;
      } else {
        continuationSuffix = allowMutationTools ? defaultContinue : readOnlyContinue;
      }
      const namesAfter = toolLogs.map((t) => t.call.tool);
      if (prefersRelay && agentSlotIndexRef.current >= 0) {
        setMessages((prev) => {
          const next = [...prev];
          const idx = agentSlotIndexRef.current;
          if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
            next[idx] = { ...next[idx], toolsUsed: namesAfter };
          }
          return next;
        });
      }
      convo.push({ role: "assistant", content: reply });
      convo.push({
        role: "tool",
        name: effectiveToolCall.tool,
        content: JSON.stringify(execResult, null, 2)
      });
    }
    throw new Error("工具调用达到上限，请缩小问题范围后重试。");
  };

  const applyPreviewResponseToUi = (preview, source) => {
    const cs = preview.change_summary && typeof preview.change_summary === "object" ? preview.change_summary : null;
    const changedCount = Array.isArray(cs?.paths) ? cs.paths.length : 0;
    const hasApprovalToken = !!preview?.approval_token;
    if (!hasApprovalToken || changedCount <= 0 || preview?.need_approval === false) {
      setApprovalOpen(false);
      setApprovalChangeSummary(null);
      setApprovalToken("");
      setApprovalRunId("default");
      setApprovalRequestHash("");
      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          content: "当前没有检测到需要写入的参数改动。是否需要我提供一版参数修改方案？"
        }
      ]);
      toast("未检测到实际参数变更，已跳过参数写入流程。", "success");
      return;
    }
    setApprovalChangeSummary(cs);
    setApprovalToken(preview.approval_token || "");
    setApprovalRunId(String(preview.run_id || "default"));
    setApprovalRequestHash(String(preview.request_hash || ""));
    setApprovalOpen(true);
    setMessages((prev) => [
      ...prev,
      { _messageId: nextMessageId(), role: "agent", content: formatChangeSummaryForChat(cs), kind: "config_summary" }
    ]);
    toast(
      source === "tool"
        ? "已通过工具触发审批预览，可在对话区「批准修改训练配置」中采纳。"
        : source === "propose"
          ? "已从 propose_patch 生成审批预览，可在对话区「批准修改训练配置」中采纳。"
          : "已生成配置 patch 预览，可在对话区「批准修改训练配置」中采纳或让 Agent 执行。",
      "success"
    );
  };

  /** 从工具链中的 agent.preview_patch 结果同步审批票据（与仅解析回复 Markdown 互补） */
  const syncApprovalFromToolLogs = (toolLogs) => {
    if (!Array.isArray(toolLogs) || !toolLogs.length) return false;
    for (let i = toolLogs.length - 1; i >= 0; i -= 1) {
      const { call, result } = toolLogs[i];
      if (call?.tool !== "agent.preview_patch" || !result) continue;
      if (hasConfigMutationAfter(toolLogs, i)) {
        continue;
      }
      const tok = result.approval_token;
      if (!tok) continue;
      if (result.status && result.status !== "ok") continue;
      applyPreviewResponseToUi(result, "tool");
      return true;
    }
    return false;
  };

  /**
   * agent.propose_patch 只返回建议 patch，不签发审批令牌；此处代为调用 /api/agent/patch/preview。
   * 工具响应形态：{ status, tool, result: { goal, patch, need_approval } }
   */
  const syncProposePatchViaPreview = async (toolLogs) => {
    if (!Array.isArray(toolLogs) || !toolLogs.length) return false;
    for (let i = toolLogs.length - 1; i >= 0; i -= 1) {
      const { call, result } = toolLogs[i];
      if (call?.tool !== "agent.propose_patch" || !result) continue;
      if (hasConfigMutationAfter(toolLogs, i)) continue;
      // 与 preview_patch 不同，部分中继响应可能省略 status；仅在明确失败时跳过
      if (result.status && result.status !== "ok") {
        continue;
      }
      const inner = result.result && typeof result.result === "object" ? result.result : result;
      const patch = inner && typeof inner === "object" && !Array.isArray(inner) ? inner.patch : null;
      if (!patch || typeof patch !== "object" || Array.isArray(patch) || !Object.keys(patch).length) {
        continue;
      }
      try {
        const preview = await previewAgentPatch({
          patch,
          run_id: (call.args && call.args.run_id) || "default",
          operator: "agent",
          reason: "agent.propose_patch"
        });
        applyPreviewResponseToUi(preview, "propose");
        return true;
      } catch (error) {
        toast(`无法从 propose_patch 生成审批预览: ${error.message}`, "error");
        return false;
      }
    }
    return false;
  };

  const maybeHandlePatch = async (result, replyText) => {
    const replyRaw = typeof replyText === "string" ? replyText : "";
    const transcriptEchoLike =
      /^\s*\[assistant\]/.test(replyRaw) ||
      /^\s*\[user\]/.test(replyRaw) ||
      /^\s*\[tool:[^\]]+\]/m.test(replyRaw) ||
      /\n\[tool:[^\]]+\]/.test(replyRaw);
    if (transcriptEchoLike) {
      return;
    }
    const patch = extractPatchFromResult(result, replyText);
    if (!patch) return;
    try {
      const preview = await previewAgentPatch({ patch });
      applyPreviewResponseToUi(preview, "markdown");
    } catch (error) {
      setMessages((prev) => [...prev, { _messageId: nextMessageId(), role: "agent", content: `Patch 校验失败: ${error.message}` }]);
      toast(`Patch 校验失败: ${error.message}`, "error");
    }
  };

  const executeUserText = async (text, clearInput = false) => {
    if (!text) return;
    if (!apiUrl.trim()) {
      toast("请先填写 Agent API 地址", "warning");
      return;
    }
    lastUserMutationIntentRef.current = isMutationIntentRequested(text);
    // 新消息到来时先清空旧审批态，避免残留 token 被误执行；若本轮确实需要审批，会在 preview 返回后重新打开。
    setApprovalOpen(false);
    setApprovalChangeSummary(null);
    setApprovalToken("");
    setApprovalRunId("default");
    setApprovalRequestHash("");
    const explicitToolCallForUserText = extractToolCallFromText(text);
    const explicitMutationToolForUserText =
      !!explicitToolCallForUserText &&
      new Set(["agent.propose_patch", "agent.preview_patch", "agent.apply_patch_with_approval", "agent.rollback_run_config"]).has(
        String(explicitToolCallForUserText.tool || "")
      );
    const hasRecentMutationContext = messages
      .slice(-8)
      .some(
        (m) =>
          m?.role === "agent" &&
          Array.isArray(m?.traceRounds) &&
          m.traceRounds.some((r) =>
            ["agent.propose_patch", "agent.preview_patch", "agent.apply_patch_with_approval", "agent.rollback_run_config"].includes(
              String(r?.tool?.name || "")
            )
          )
      );
    const continuationMutationIntent = hasRecentMutationContext && isContinuationLikeMutationText(text);
    const hasRecentAnyToolContext = messages
      .slice(-8)
      .some(
        (m) =>
          m?.role === "agent" &&
          Array.isArray(m?.traceRounds) &&
          m.traceRounds.some((r) => !!r?.tool?.name || (Array.isArray(r?.jsonCodeBlocks) && r.jsonCodeBlocks.length > 0))
      );
    const allowMutationForThisTurn =
      isMutationIntentRequested(text) || explicitMutationToolForUserText || continuationMutationIntent;
    setLoading(true);
    if (clearInput) setInput("");
    setMessages((prev) => {
      return [...prev, { _messageId: nextMessageId(), role: "user", content: text }];
    });
    const turnStartTs = Date.now();
    const preTail = messages.slice(-5);
    const preHasAnyStreamingAgent = messages.some((m) => m?.role === "agent" && !!m?.streaming);
    const preLatest = messages.length ? messages[messages.length - 1] : null;
    try {
      const { payload, reply, displayReply, displayReasoning, target, toolLogs, streamedRelay, traceRounds } =
        await runAgentWithTools({
          userText: text,
          forceAllowMutationTools: allowMutationForThisTurn,
          turnStartTs
        });
      const data = payload;
      if (!streamedRelay) {
        const names = toolLogs.map((t) => t.call.tool);
        setMessages((prev) => [
          ...prev,
          {
            _messageId: nextMessageId(),
            role: "agent",
            content: displayReply,
            toolsUsed: names,
            modelName: resolveModelName(apiModel),
            ...(displayReasoning ? { reasoningApi: displayReasoning } : {}),
            ...(traceRounds?.length ? { traceRounds } : {})
          }
        ]);
      }
      if (target.kind === "openai") {
        toast("已通过 OpenAI 兼容接口完成请求", "success");
      } else if (target.kind === "backend-relay") {
        toast("已通过本地中转完成请求（已绕过浏览器跨域）", "success");
      }
      /* 已成功 apply/rollback 时：不得再跑 syncApprovalFromToolLogs，否则会命中同一条 toolLogs 里更早的 preview_patch，再次打开对话区审批（死循环） */
      const configMutationDone = toolLogs.some(
        (t) =>
          (t.call?.tool === "agent.apply_patch_with_approval" || t.call?.tool === "agent.rollback_run_config") &&
          t.result?.status === "ok"
      );
      if (configMutationDone) {
        for (let i = toolLogs.length - 1; i >= 0; i -= 1) {
          const { call, result } = toolLogs[i];
          if (!result || result.status !== "ok") continue;
          if (call?.tool === "agent.rollback_run_config" && result.config && typeof result.config === "object") {
            const mt =
              typeof result.file_mtime_ns === "number" ? { file_mtime_ns: result.file_mtime_ns } : {};
            broadcastDistillConfigUpdate(result.config, "agent-chat-rollback", mt);
            break;
          }
        }
      }
      if (!configMutationDone) {
        if (!allowMutationForThisTurn) {
          const hasPatchLikeOutput = !!extractPatchFromResult(data, reply);
          const hasMutationToolInLogs = toolLogs.some((t) =>
            ["agent.propose_patch", "agent.preview_patch", "agent.apply_patch_with_approval", "agent.rollback_run_config"].includes(
              String(t.call?.tool || "")
            )
          );
          if (hasPatchLikeOutput || hasMutationToolInLogs) {
            setMessages((prev) => [
              ...prev,
              {
                _messageId: nextMessageId(),
                role: "agent",
                content: "当前请求未明确要求修改参数，已跳过配置写入。是否需要我提供一版参数修改方案？"
              }
            ]);
          }
        } else if (!syncApprovalFromToolLogs(toolLogs)) {
          if (!(await syncProposePatchViaPreview(toolLogs))) {
            await maybeHandlePatch(data, reply);
          }
        }
      }
    } catch (error) {
      setMessages((prev) => [...prev, { _messageId: nextMessageId(), role: "agent", content: `请求失败: ${error.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const send = async () => {
    const text = input.trim();
    await executeUserText(text, true);
  };

  const sendPresetMessage = async (presetText) => {
    const text = (presetText || "").trim();
    const panelEl = document.getElementById("panel-agent");
    const rootEl = document.getElementById("root");
    Promise.resolve().then(() => {
      const panelAfter = document.getElementById("panel-agent");
      const rootAfter = document.getElementById("root");
    });
    // #endregion
    await executeUserText(text, false);
  };

  const testAgentApi = async () => {
    if (!apiUrl.trim()) return toast("请先填写 Agent API 地址", "warning");
    if (isArkApiUrl(apiUrl) && !apiModel.trim()) {
      toast("检测到方舟地址，请先填写「模型名 / Endpoint ID」（如 ep-xxxxxx）", "warning");
      return;
    }
    try {
      const { target } = await requestAgentWithFallback({
        apiUrl,
        apiKey,
        apiModel,
        modelName: resolveModelName(apiModel),
        text: "ping",
        mode: "test",
        systemPrompt: ""
      });
      toast(`Agent API 连接成功（${target.kind}）`, "success");
    } catch (error) {
      toast(`Agent API 连接失败: ${error.message}`, "error");
    }
  };

  const sendAgentExecuteApproval = async () => {
    if (!lastUserMutationIntentRef.current) {
      setApprovalOpen(false);
      setApprovalChangeSummary(null);
      setApprovalToken("");
      setApprovalRunId("default");
      setApprovalRequestHash("");
      toast("当前不是参数修改请求，已阻止执行残留审批票据。", "warning");
      return;
    }
    if (!approvalToken) {
      toast("暂无待执行的审批票据", "warning");
      return;
    }
    try {
      const rawExecResult = await executeAgentTool({
        tool: "agent.apply_patch_with_approval",
        args: {
          approval_token: approvalToken,
          run_id: approvalRunId || "default",
          request_hash: approvalRequestHash || undefined,
          operator: "agent-ui"
        }
      });
      const execResult = normalizeToolExecutionResult(rawExecResult);
      setApprovalOpen(false);
      setApprovalChangeSummary(null);
      setApprovalToken("");
      setApprovalRunId("default");
      setApprovalRequestHash("");
      toast("已执行 agent.apply_patch_with_approval", "success");
      let cfgToBroadcast = execResult?.config;
      let mtimeExtra = {};
      if (typeof execResult?.file_mtime_ns === "number") {
        mtimeExtra = { file_mtime_ns: execResult.file_mtime_ns };
      }
      if (!cfgToBroadcast || typeof cfgToBroadcast !== "object") {
        try {
          const data = await fetchDistillConfig();
          cfgToBroadcast = data?.config;
          if (typeof data?.file_mtime_ns === "number") {
            mtimeExtra = { file_mtime_ns: data.file_mtime_ns };
          }
        } catch {
          cfgToBroadcast = null;
        }
      }
      if (cfgToBroadcast && typeof cfgToBroadcast === "object") {
        broadcastDistillConfigUpdate(cfgToBroadcast, "agent-ui-apply", mtimeExtra);
      }
    } catch (error) {
      toast(`执行失败: ${error.message}`, "error");
    }
  };

  return (
    <AgentPanelView
      active={active}
      apiUrl={apiUrl}
      setApiUrl={setApiUrl}
      apiKey={apiKey}
      setApiKey={setApiKey}
      apiModel={apiModel}
      setApiModel={setApiModel}
      loading={loading}
      saveConfig={saveConfig}
      testAgentApi={testAgentApi}
      approvalOpen={approvalOpen}
      approvalChangeSummary={approvalChangeSummary}
      onCloseApproval={() => {
        setApprovalOpen(false);
        setApprovalChangeSummary(null);
      }}
      sendAgentExecuteApproval={sendAgentExecuteApproval}
      approvalToken={approvalToken}
      sendPresetMessage={sendPresetMessage}
      messages={messages}
      onClearMessages={() => setMessages([])}
      onExportSession={exportAgentSession}
      relayReasoningHint={!/^https?:\/\//i.test(String(apiUrl || "").trim())}
      chatMessagesRef={chatMessagesRef}
      chatTextareaRef={chatTextareaRef}
      input={input}
      setInput={setInput}
      resizeChatInput={resizeChatInput}
      send={send}
    />
  );
}

export default AgentPanel;
