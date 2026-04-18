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

function loadStoredAgentMessages() {
  try {
    const raw = window.localStorage.getItem(AGENT_CHAT_STORAGE_KEY);
    if (!raw) return [];
    const p = JSON.parse(raw);
    return Array.isArray(p) ? p : [];
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
  const agentSlotIndexRef = useRef(-1);
  const chatStickToBottomRef = useRef(true);
  const chatScrollRafRef = useRef(0);

  const scheduleScrollRef = useRef(() => {});
  scheduleScrollRef.current = () => {
    const el = chatMessagesRef.current;
    if (!el || !active) return;
    if (!chatStickToBottomRef.current) return;
    if (chatScrollRafRef.current) return;
    chatScrollRafRef.current = window.requestAnimationFrame(() => {
      chatScrollRafRef.current = 0;
      const box = chatMessagesRef.current;
      if (!box || !active || !chatStickToBottomRef.current) return;
      box.scrollTop = box.scrollHeight;
    });
  };

  useLayoutEffect(() => {
    if (!active) return;
    scheduleScrollRef.current();
  }, [messages, loading, active]);

  useEffect(() => {
    const el = chatMessagesRef.current;
    if (!el) return;
    const onScroll = () => {
      const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
      chatStickToBottomRef.current = dist < 100;
    };
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [active]);

  useEffect(() => {
    const el = chatMessagesRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      if (!active) return;
      scheduleScrollRef.current();
    });
    ro.observe(el);
    return () => ro.disconnect();
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
    const maxHeight = lineHeight * 3 + paddingTop + paddingBottom + borderTop + borderBottom;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
    el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
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
      "**修改 distill 配置（configs/distill_config.yaml）时**：优先输出 `{\"tool\":\"agent.preview_patch\",\"args\":{\"patch\":{...}}}`（可先 `agent.validate_patch`）；若只调用了 `agent.propose_patch`，界面也会自动用返回的 patch 请求预览并在对话区显示「批准修改训练配置」。**禁止**在最终答复里用「是否需要我执行/是否生成补丁」等话术向用户索要确认。",
      "**`patch` 必须与审批内容严格一致**：只包含你打算改动的顶层段（distillation / training / output）及其字段；**禁止**在 patch 里附带未在答复中说明的参数（例如只讨论 kd loss 却写入 temperature）。自然语言列出的每一项都必须在 patch 中出现且数值一致。",
      "在输出 tool JSON 前后，用一两句话概括拟修改的字段与意图；**字段级旧值→新值以对话区审批摘要与聊天中的服务端摘要为准**，避免与 patch 不一致。",
      "终端里的训练命令（```powershell``` / ```bash```）只能作为补充说明；真正写入配置必须经过上述工具链或界面审批。",
      "当不再需要工具、输出最终答复时：用自然语言说明变更理由与风险；若已调用 `agent.preview_patch`，只需提示用户在对话区「批准修改训练配置」中批准，不要重复询问。仅在确定无法走工具链时，才用 JSON 代码块给出结构化 patch 作为兜底。",
      "**审批/写入流程结束后**：若用户未主动要求继续改配置，你**不得**主动提出新的修改建议、不得追问「要不要继续调整」「是否还要改某参数」「需不需要再预览一版」等；**不得**自动再发起 `agent.propose_patch` / `agent.preview_patch` 或引导用户进入下一轮审批。此时只做简短收尾（例如已写入/已按对话区审批操作即可训练），然后停止。",
      "**仅当用户明确说出**要改配置、改某字段、再优化、再出一版 patch 等意图时，你才可以再次使用配置相关工具或给出修改建议。"
    ].join("\n");
  };

  const runAgentWithTools = async ({ userText, maxRounds = 6 }) => {
    agentSlotIndexRef.current = -1;
    const systemPrompt = await buildAgentSystemPrompt();
    const convo = [...messages, { role: "user", content: userText }];
    const toolLogs = [];
    const traceRounds = [];
    const allowMutationTools = /修改|调参|调整|优化|patch|preview|采纳|批准|写入|apply|执行|变更|改配置|再来一版|继续改/i.test(
      String(userText || "")
    );
    const mutationTools = new Set([
      "agent.propose_patch",
      "agent.preview_patch",
      "agent.apply_patch_with_approval",
      "agent.rollback_run_config"
    ]);
    const canAutoBridgeFromAnalysis = () => {
      if (!allowMutationTools) return false;
      const hasAnalyze = toolLogs.some((t) => t.call?.tool === "agent.analyze_params");
      if (!hasAnalyze) return false;
      const hasPatchFlow = toolLogs.some((t) => mutationTools.has(t.call?.tool));
      return !hasPatchFlow;
    };
    const prefersRelayGlobal = /^https?:\/\//i.test(apiUrl.trim());
    const defaultContinue = "请继续。若需工具则按约定输出 tool JSON。";
    /** 写入/回滚完成后若仍用「请继续+tool JSON」，模型会再出一轮 patch，形成审批死循环 */
    const finalizeAfterMutation =
      "配置变更已通过工具落盘。请仅用一两句自然语言确认完成（不要输出 JSON 代码块、不要调用任何工具）。在用户未明确要求继续改配置前，禁止主动提出修改建议、禁止追问是否继续调整、禁止再次调用 propose_patch / preview_patch / apply_patch / rollback 相关工具。";
    let continuationSuffix = defaultContinue;

    const flushTrace = (reply, reasoning, toolInfo, execRes) => {
      traceRounds.push({
        round: traceRounds.length + 1,
        reply,
        reasoning: reasoning || "",
        tool: toolInfo ? { name: toolInfo.tool, args: toolInfo.args || {} } : null,
        toolResultSummary:
          execRes !== undefined && execRes !== null
            ? summarizeToolResultForTrace(toolInfo?.tool, execRes)
            : undefined
      });
      if (prefersRelayGlobal && agentSlotIndexRef.current >= 0) {
        setMessages((prev) => {
          const next = [...prev];
          const idx = agentSlotIndexRef.current;
          if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
            next[idx] = { ...next[idx], traceRounds: [...traceRounds] };
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

      if (prefersRelay && round === 0) {
        setMessages((prev) => {
          const idx = prev.length;
          agentSlotIndexRef.current = idx;
          return [...prev, { role: "agent", content: "", reasoningApi: "", toolsUsed: [], streaming: true }];
        });
      } else if (prefersRelay && round > 0) {
        setMessages((prev) => {
          const next = [...prev];
          const idx = agentSlotIndexRef.current;
          if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
            next[idx] = { ...next[idx], streaming: true };
          }
          return next;
        });
      }

      const onDelta = prefersRelay
        ? ({ reply, reasoning }) => {
            setMessages((prev) => {
              const next = [...prev];
              const idx = agentSlotIndexRef.current;
              if (idx >= 0 && idx < next.length && next[idx].role === "agent" && next[idx].streaming) {
                next[idx] = {
                  ...next[idx],
                  content: reply || "",
                  reasoningApi: reasoning || "",
                  streaming: true
                };
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
          modelName: resolveModelName(apiModel),
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
        if (typeof payload?.reasoning === "string" && payload.reasoning.trim() && prefersRelay) {
          setMessages((prev) => {
            const next = [...prev];
            const idx = agentSlotIndexRef.current;
            if (idx >= 0 && idx < next.length && next[idx].role === "agent" && next[idx].streaming) {
              next[idx] = { ...next[idx], reasoningApi: payload.reasoning };
            }
            return next;
          });
        }
      } finally {
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
      }

      const toolNames = toolLogs.map((t) => t.call.tool);
      if (prefersRelay && target?.kind === "backend-relay") {
        setMessages((prev) => {
          const next = [...prev];
          const idx = agentSlotIndexRef.current;
          if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
            const mergedReasoning =
              displayReasoning && String(displayReasoning).trim()
                ? displayReasoning
                : next[idx].reasoningApi || "";
            next[idx] = {
              ...next[idx],
              content: displayReply,
              toolsUsed: toolNames,
              ...(mergedReasoning && String(mergedReasoning).trim() ? { reasoningApi: mergedReasoning } : {})
            };
          }
          return next;
        });
      }

      const toolCall = extractToolCallFromText(reply);
      if (!toolCall) {
        if (canAutoBridgeFromAnalysis()) {
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
          const execResult = await executeAgentTool({ tool: fallbackCall.tool, args: fallbackCall.args });
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
          continuationSuffix = defaultContinue;
          flushTrace(displayReply, displayReasoning, fallbackCall, execResult);
          continue;
        }
        flushTrace(displayReply, displayReasoning, null, null);
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
      if (mutationTools.has(toolCall.tool) && !allowMutationTools) {
        flushTrace(displayReply, displayReasoning, toolCall, { blocked: true, reason: "mutation_not_allowed" });
        return {
          payload,
          reply,
          displayReply:
            "当前消息未明确要求修改配置，已阻止自动变更建议与审批流程。若需要修改，请明确说明“请修改哪些参数/请生成并预览 patch”。",
          displayReasoning,
          target,
          toolLogs,
          streamedRelay: target?.kind === "backend-relay",
          traceRounds
        };
      }
      if (toolCall.tool === "agent.apply_patch_with_approval") {
        flushTrace(displayReply, displayReasoning, toolCall, { blocked: true, reason: "apply_via_sidebar" });
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
      const execResult = await executeAgentTool({ tool: toolCall.tool, args: toolCall.args || {} });
      toolLogs.push({ call: toolCall, result: execResult });
      flushTrace(displayReply, displayReasoning, toolCall, execResult);
      if (toolCall.tool === "agent.apply_patch_with_approval" && execResult && execResult.status === "ok") {
        continuationSuffix = finalizeAfterMutation;
      } else if (toolCall.tool === "agent.rollback_run_config" && execResult && execResult.status === "ok") {
        continuationSuffix = finalizeAfterMutation;
      } else {
        continuationSuffix = defaultContinue;
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
        name: toolCall.tool,
        content: JSON.stringify(execResult, null, 2)
      });
    }
    throw new Error("工具调用达到上限，请缩小问题范围后重试。");
  };

  const applyPreviewResponseToUi = (preview, source) => {
    const cs = preview.change_summary && typeof preview.change_summary === "object" ? preview.change_summary : null;
    setApprovalChangeSummary(cs);
    setApprovalToken(preview.approval_token || "");
    setApprovalRunId(String(preview.run_id || "default"));
    setApprovalRequestHash(String(preview.request_hash || ""));
    setApprovalOpen(true);
    setMessages((prev) => [...prev, { role: "agent", content: formatChangeSummaryForChat(cs), kind: "config_summary" }]);
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
      setMessages((prev) => [...prev, { role: "agent", content: `Patch 校验失败: ${error.message}` }]);
      toast(`Patch 校验失败: ${error.message}`, "error");
    }
  };

  const executeUserText = async (text, clearInput = false) => {
    if (!text) return;
    if (!apiUrl.trim()) {
      toast("请先填写 Agent API 地址", "warning");
      return;
    }
    chatStickToBottomRef.current = true;
    setLoading(true);
    if (clearInput) setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    try {
      const { payload, reply, displayReply, displayReasoning, target, toolLogs, streamedRelay, traceRounds } =
        await runAgentWithTools({
          userText: text
        });
      const data = payload;
      if (!streamedRelay) {
        const names = toolLogs.map((t) => t.call.tool);
        setMessages((prev) => [
          ...prev,
          {
            role: "agent",
            content: displayReply,
            toolsUsed: names,
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
        if (!syncApprovalFromToolLogs(toolLogs)) {
          if (!(await syncProposePatchViaPreview(toolLogs))) {
            await maybeHandlePatch(data, reply);
          }
        }
      }
    } catch (error) {
      setMessages((prev) => [...prev, { role: "agent", content: `请求失败: ${error.message}` }]);
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
      toast("检测到方舟地址，请先填写“模型名 / Endpoint ID”（如 ep-xxxxxx）", "warning");
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
    if (!approvalToken) {
      toast("暂无待执行的审批票据", "warning");
      return;
    }
    try {
      const execResult = await executeAgentTool({
        tool: "agent.apply_patch_with_approval",
        args: {
          approval_token: approvalToken,
          run_id: approvalRunId || "default",
          request_hash: approvalRequestHash || undefined,
          operator: "agent-ui"
        }
      });
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
