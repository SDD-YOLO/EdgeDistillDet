import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { executeAgentTool, fetchAgentPrompts, fetchAgentTools, previewAgentPatch } from "../../api/agentApi";
import { fetchDistillConfig } from "../../api/configApi";
import { broadcastDistillConfigUpdate } from "../../constants/distillConfigSync";
import AgentPanelView from "./AgentPanelView";
import { buildDisplayReplyAndReasoning, extractReplyAndReasoningFromPayload } from "../../utils/agentPayload";
import {
  extractPatchFromResult,
  hasConfigMutationAfter,
  summarizeToolResultForTrace
} from "../../utils/patchHelpers";
import { extractToolCallFromText } from "../../utils/toolHelpers";
import { requestAgentWithFallback } from "./agentRuntime";
import { sanitizeBlockedCommandHints } from "./utils/agentTextUtils";


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

  const buildAgentSystemPrompt = async (_userText = "") => {
    let contract = null;
    let prompts = null;
    try {
      [contract, prompts] = await Promise.all([fetchAgentTools(), fetchAgentPrompts()]);
    } catch {
      try {
        contract = await fetchAgentTools();
      } catch {
        contract = null;
      }
      prompts = null;
    }
    const toolList = Array.isArray(contract?.tools)
      ? contract.tools.map((t) => `- ${t.name}: input=${JSON.stringify(t.input || {})}, output=${t.output || ""}`).join("\n")
      : "- agent.get_context\n- agent.preview_patch\n- agent.apply_patch_with_approval\n- agent.list_run_history\n- agent.rollback_run_config";
    const yamlPrompt = typeof prompts?.chat_system_prompt === "string" ? prompts.chat_system_prompt : "";
    if (yamlPrompt.trim()) {
      return yamlPrompt.replaceAll("{{tool_list}}", toolList);
    }
    return [
      "你是训练参数优化 Agent，专注于 YOLO 蒸馏训练场景。",
      "## 可用工具",
      toolList
    ].join("\n");
  };

  const runAgentWithTools = async ({
    userText,
    maxRounds = 6,
    forceAllowMutationTools = false,
    turnStartTs = Date.now()
  }) => {
    agentSlotIndexRef.current = -1;
    let missingSlotLogged = false;
    const systemPrompt = await buildAgentSystemPrompt(userText);
    const recentMessages = Array.isArray(messages) ? messages.slice(-4) : [];
    const convo = [...recentMessages, { role: "user", content: userText }];
    const toolLogs = [];
    const traceRounds = [];
    const prefersRelayGlobal = /^https?:\/\//i.test(apiUrl.trim());
    const transcriptText = convo
      .map((m) => {
        if (m.role === "tool") return `[tool:${m.name}]\n${m.content}`;
        return `[${m.role}] ${m.content}`;
      })
      .join("\n\n");
    const prompt = transcriptText;
    const prefersRelay = prefersRelayGlobal;
    const resolvedModelName = resolveModelName(apiModel);

    if (prefersRelay) {
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
    }

    const roundMap = new Map();
    const upsertRound = (step) => {
      const key = Number(step) || roundMap.size + 1;
      if (!roundMap.has(key)) {
        roundMap.set(key, {
          round: key,
          reply: "",
          reasoning: "",
          jsonCodeBlocks: [],
          tool: null
        });
      }
      return roundMap.get(key);
    };

    const onDelta = prefersRelay
      ? ({ reply, reasoning, event }) => {
          if (event?.event_type === "model_output") {
            const step = event.step || 1;
            const payload = event.payload && typeof event.payload === "object" ? event.payload : {};
            const row = upsertRound(step);
            row.reply = String(payload.reply || reply || "");
            row.reasoning = String(payload.reasoning || reasoning || "");
            row.jsonCodeBlocks = extractJsonCodeBlocks(row.reply);
          } else if (event?.event_type === "tool_end") {
            const step = event.step || 1;
            const payload = event.payload && typeof event.payload === "object" ? event.payload : {};
            const row = upsertRound(step);
            row.tool = { name: String(payload.tool || ""), args: payload.args || {} };
            row.toolResultSummary = summarizeToolResultForTrace(payload.tool, payload.result);
            toolLogs.push({
              call: { tool: String(payload.tool || ""), args: payload.args || {} },
              result: normalizeToolExecutionResult(payload.result)
            });
          } else if (event?.event_type === "retrieval_hit") {
            const step = event.step || 1;
            const payload = event.payload && typeof event.payload === "object" ? event.payload : {};
            const row = upsertRound(step);
            const retrievalJson = JSON.stringify({ retrieval: payload }, null, 2);
            if (!row.jsonCodeBlocks.includes(retrievalJson)) {
              row.jsonCodeBlocks.push(retrievalJson);
            }
          }
          const traceForUi = Array.from(roundMap.values()).sort((a, b) => a.round - b.round);
          setMessages((prev) => {
            const next = [...prev];
            const idx = agentSlotIndexRef.current;
            if (idx >= 0 && idx < next.length && next[idx].role === "agent" && next[idx].streaming) {
              next[idx] = {
                ...next[idx],
                content: reply || "",
                reasoningApi: reasoning || "",
                streaming: true,
                traceRounds: traceForUi,
                traceOpen: traceForUi.some((r) => !!r.tool?.name),
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
        onRelayFallback,
        runId: "default",
        sessionId: `agent-${turnStartTs}`,
        ragOptions: { mode: "hybrid", top_k: 5 },
        toolPolicy: {
          allow_mutation_tools: true,
          allow_auto_apply: false
        },
        maxSteps: Math.max(2, Number(maxRounds || 6))
      });
      payload = result.payload;
      target = result.target;
      const parsed = extractReplyAndReasoningFromPayload(payload?.reply || payload);
      reply = parsed.reply;
      const built = buildDisplayReplyAndReasoning(reply, parsed.reasoning);
      displayReply = built.displayReply;
      displayReasoning = built.displayReasoning;
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

    const finalTraceRounds = Array.from(roundMap.values()).sort((a, b) => a.round - b.round);
    const toolNames = toolLogs.map((t) => t.call.tool).filter(Boolean);

    if (prefersRelay && target?.kind === "backend-relay") {
      setMessages((prev) => {
        const next = [...prev];
        const idx = agentSlotIndexRef.current;
        if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
          next[idx] = {
            ...next[idx],
            content: displayReply,
            toolsUsed: toolNames,
            traceRounds: finalTraceRounds,
            traceOpen: finalTraceRounds.some((r) => !!r.tool?.name),
            modelName: payload?.model || resolvedModelName,
            streaming: false,
            reasoningApi: displayReasoning || ""
          };
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
      traceRounds: finalTraceRounds
    };
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
    toast(
      source === "tool"
        ? "已通过工具触发审批预览，可在对话区「批准修改训练配置」中采纳。"
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
    // 语义驱动：不再基于关键词预判是否“改参意图”，由模型在工具链中自行决策是否进入 preview/apply 审批流程。
    lastUserMutationIntentRef.current = true;
    // 新消息到来时先清空旧审批态，避免残留 token 被误执行；若本轮确实需要审批，会在 preview 返回后重新打开。
    setApprovalOpen(false);
    setApprovalChangeSummary(null);
    setApprovalToken("");
    setApprovalRunId("default");
    setApprovalRequestHash("");
    // 始终允许模型在理解语义后自主选择是否调用 mutation tools；
    // 实际写入仍受 preview+approval 双重门控约束。
    const allowMutationForThisTurn = true;
    setLoading(true);
    if (clearInput) setInput("");
    setMessages((prev) => {
      return [...prev, { _messageId: nextMessageId(), role: "user", content: text }];
    });
    const turnStartTs = Date.now();
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
            ["agent.preview_patch", "agent.apply_patch_with_approval", "agent.rollback_run_config"].includes(
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
          await maybeHandlePatch(data, reply);
        }
        if (!configMutationDone && allowMutationForThisTurn) {
          const replyText = String(reply || "");
          const approvalCta = /批准修改训练配置|在对话区[「"']*批准|审批区|让\s*agent\s*执行|去审批|请.*批准|核对以下变更/.test(replyText);
          const hasPreviewInLogs = toolLogs.some((t) => t.call?.tool === "agent.preview_patch");
          if (approvalCta && !hasPreviewInLogs) {
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
