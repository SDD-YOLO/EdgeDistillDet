import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { TextField } from "../../components/forms/TextField";
import { Button } from "../../components/ui/button";
import { AGENT_QUICK_PROMPTS, sanitizeBlockedCommandHints, softenAgentBubbleText } from "./agentHelpers";

function MarkdownText({ text, className = "" }) {
  const value = String(text || "");
  return (
    <ReactMarkdown
      className={`agent-markdown ${className}`.trim()}
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <p className="agent-md-p">{children}</p>,
        ul: ({ children }) => <ul className="agent-md-list">{children}</ul>,
        ol: ({ children }) => <ol className="agent-md-list">{children}</ol>,
        li: ({ children }) => <li className="agent-md-li">{children}</li>,
        code: ({ inline, children, ...props }) =>
          inline ? (
            <code className="agent-md-inline-code" {...props}>
              {children}
            </code>
          ) : (
            <pre className="agent-md-code-block">
              <code {...props}>{children}</code>
            </pre>
          )
      }}
    >
      {value}
    </ReactMarkdown>
  );
}

function ChatBubbleBody({
  role,
  content,
  modelName,
  reasoningApi,
  toolsUsed,
  streaming,
  messageKind,
  traceRounds,
  traceOpen,
  relayMayMissReasoning,
  bubbleIndex,
  isLatestAgentBubble
}) {
  const bubbleRef = useRef(null);
  const traceListRef = useRef(null);
  const traceAutoFollowRef = useRef(true);
  const [traceExpanded, setTraceExpanded] = useState(true);

  const sanitizeTraceText = (value) =>
    sanitizeBlockedCommandHints(String(value || ""))
      .replace(/\r\n/g, "\n")
      .replace(/(^|\n)\s*\[tool:[^\]\n]+\]\s*/gi, "$1")
      .replace(/(^|\n)\s*\[(assistant|agent)\][^\n]*/gi, "$1")
      .replace(/\[(assistant|agent)\]\s*/gi, "")
      .replace(/\n{3,}/g, "\n\n")
      .trim();

  const rawText = content ?? "";
  const text =
    role === "agent" && messageKind !== "config_summary"
      ? softenAgentBubbleText(rawText, !!streaming)
      : rawText;
  const tools = Array.isArray(toolsUsed)
    ? toolsUsed.filter(Boolean)
    : [];
  const rounds = Array.isArray(traceRounds) ? traceRounds : [];
  const modelLabel = String(modelName || "").trim();

  // 仅显示真实工具回合：存在工具名或存在 JSON 代码块
  const toolRounds = rounds.filter(
    (r) => !!r?.tool?.name || (Array.isArray(r?.jsonCodeBlocks) && r.jsonCodeBlocks.length > 0)
  );
  const hasToolRounds = toolRounds.length > 0;
  const showStreamingPlaceholder = role === "agent" && !!streaming && !hasToolRounds && !String(text || "").trim();
  const showToolStreamingPlaceholder = role === "agent" && !!streaming && hasToolRounds && !String(text || "").trim();

  useEffect(() => {
    if (role !== "agent") return;
    if (!hasToolRounds) return;
    const listEl = traceListRef.current;
    if (!listEl) return;
    traceAutoFollowRef.current = true;
    const onTraceScroll = () => {
      const distance = listEl.scrollHeight - listEl.scrollTop - listEl.clientHeight;
      traceAutoFollowRef.current = distance < 60;
    };
    listEl.addEventListener("scroll", onTraceScroll, { passive: true });
    return () => listEl.removeEventListener("scroll", onTraceScroll);
  }, [role, hasToolRounds, bubbleIndex]);

  useEffect(() => {
    if (role !== "agent") return;
    if (!hasToolRounds) return;
    const listEl = traceListRef.current;
    if (!listEl) return;
    const distanceFromBottom = listEl.scrollHeight - listEl.scrollTop - listEl.clientHeight;
    const isNearBottom = distanceFromBottom < 60;
    const shouldAutoFollow = isNearBottom || traceAutoFollowRef.current;
    if (streaming) {
      if (shouldAutoFollow) {
        listEl.scrollTop = listEl.scrollHeight;
        traceAutoFollowRef.current = true;
      }
    } else {
      listEl.scrollTop = listEl.scrollHeight;
    }
  }, [role, hasToolRounds, streaming, toolRounds.length, bubbleIndex, traceExpanded]);

  if (role !== "agent") {
    return <div className="chat-plain chat-pre-wrap">{text}</div>;
  }

  if (messageKind === "config_summary") {
    return (
      <div className="agent-bubble-config-summary chat-pre-wrap">
        {text}
      </div>
    );
  }

  if (showStreamingPlaceholder) {
    return (
      <div className="agent-bubble-simple" ref={bubbleRef}>
        {modelLabel ? <div className="agent-model-label">模型：{modelLabel}</div> : null}
        <div className="agent-answer-content">处理中...</div>
      </div>
    );
  }

  // Show tool trace OR final answer
  if (hasToolRounds) {
    return (
      <div className="agent-bubble-nested" ref={bubbleRef}>
        {/* Outer bubble - white background for final answer */}
        <div className="agent-answer-bubble-outer">
          {modelLabel ? <div className="agent-model-label">模型：{modelLabel}</div> : null}
          {/* Tool trace details with JSON code blocks */}
          <details
            className="agent-trace-details-nested"
            open={traceExpanded}
            onToggle={(e) => {
              setTraceExpanded(e.currentTarget.open);
            }}
          >
            <summary className="agent-trace-summary-nested">
              <span className="material-icons">build_circle</span>
              <span>调用工具中…（{toolRounds.length} 轮）</span>
            </summary>
            <ol className="agent-trace-list-nested" ref={traceListRef}>
              {toolRounds.map((r, tidx) => (
                <li key={`trace-${tidx}-${r.round}`} className="agent-trace-item-nested">
                  <div className="agent-trace-round-title">第 {r.round} 轮</div>
                  <div className="agent-trace-reply-snippet">
                    {typeof r.reply === "string" ? sanitizeTraceText(r.reply).slice(0, 2000) : ""}
                  </div>
                  {Array.isArray(r.jsonCodeBlocks) && r.jsonCodeBlocks.length > 0 ? (
                    <div className="agent-trace-json-blocks">
                      <div className="agent-trace-tool-label">JSON 代码块（{r.jsonCodeBlocks.length}）</div>
                      {r.jsonCodeBlocks.map((b, bidx) => (
                        <pre key={`trace-json-${tidx}-${bidx}`} className="agent-trace-pre-nested">
                          {sanitizeTraceText(b)}
                        </pre>
                      ))}
                    </div>
                  ) : null}
                  <div className="agent-trace-tool">
                    {r.tool?.name ? (
                      <>
                        <span className="agent-trace-tool-label">工具: {r.tool.name}</span>
                        <pre className="agent-trace-pre-nested">
                          {sanitizeTraceText(JSON.stringify(r.toolResultSummary ?? {}, null, 2))}
                        </pre>
                      </>
                    ) : null}
                  </div>
                </li>
              ))}
            </ol>
          </details>

          {/* Final answer content */}
          {showToolStreamingPlaceholder ? (
            <div className="agent-answer-content-outer">处理中...</div>
          ) : null}
          {!streaming && text.trim() && (
            <div className="agent-answer-content-outer">
              <MarkdownText text={text} />
            </div>
          )}
        </div>
      </div>
    );
  }

  // Simple answer bubble (no reasoning)
  return (
    <div className="agent-bubble-simple" ref={bubbleRef}>
      {modelLabel ? <div className="agent-model-label">模型：{modelLabel}</div> : null}
      {tools.length > 0 ? (
        <div className="agent-bubble-tools" role="list" aria-label="已使用工具">
          {tools.map((name) => (
            <span key={name} className="md-chip md-chip-assist" role="listitem">
              <span className="material-icons md-chip-icon" aria-hidden>
                build
              </span>
              {name}
            </span>
          ))}
        </div>
      ) : null}
      <div className="agent-answer-content">
        <MarkdownText text={text} />
      </div>
    </div>
  );
}

function AgentPanelView({
  active,
  apiUrl,
  setApiUrl,
  apiKey,
  setApiKey,
  apiModel,
  setApiModel,
  loading,
  saveConfig,
  testAgentApi,
  approvalOpen,
  approvalChangeSummary,
  onCloseApproval,
  sendAgentExecuteApproval,
  approvalToken,
  sendPresetMessage,
  messages,
  onClearMessages,
  onExportSession = () => {},
  relayReasoningHint = false,
  chatMessagesRef,
  chatTextareaRef,
  input,
  setInput,
  resizeChatInput,
  send
}) {
  const latestAgentIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i]?.role === "agent") return i;
    }
    return -1;
  })();
  const hasTailStreamingAgent = (() => {
    const tail = Array.isArray(messages) ? messages.slice(-4) : [];
    return tail.some((m) => m?.role === "agent" && !!m?.streaming);
  })();
  const exportButtonRef = useRef(null);
  const exportIconRef = useRef(null);

  useEffect(() => {
    if (!active || !exportButtonRef.current) return;
    const button = exportButtonRef.current;
    const icon = exportIconRef.current;
    const buttonStyle = getComputedStyle(button);
    const iconStyle = icon ? getComputedStyle(icon) : null;
    const rect = button.getBoundingClientRect();

    // #region agent log
    fetch("http://127.0.0.1:7934/ingest/2c4bcf68-efd6-4fd1-8130-1f5a368246bc", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "b84065" },
      body: JSON.stringify({
        sessionId: "b84065",
        runId: "baseline",
        hypothesisId: "H1",
        location: "AgentPanelView.jsx:263",
        message: "icon button rendered width baseline",
        data: { className: button.className, width: rect.width, height: rect.height },
        timestamp: Date.now()
      })
    }).catch(() => {});
    // #endregion

    // #region agent log
    fetch("http://127.0.0.1:7934/ingest/2c4bcf68-efd6-4fd1-8130-1f5a368246bc", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "b84065" },
      body: JSON.stringify({
        sessionId: "b84065",
        runId: "baseline",
        hypothesisId: "H2",
        location: "AgentPanelView.jsx:264",
        message: "check min-width and padding from md-btn",
        data: {
          minWidth: buttonStyle.minWidth,
          paddingLeft: buttonStyle.paddingLeft,
          paddingRight: buttonStyle.paddingRight,
          width: buttonStyle.width
        },
        timestamp: Date.now()
      })
    }).catch(() => {});
    // #endregion

    // #region agent log
    fetch("http://127.0.0.1:7934/ingest/2c4bcf68-efd6-4fd1-8130-1f5a368246bc", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "b84065" },
      body: JSON.stringify({
        sessionId: "b84065",
        runId: "baseline",
        hypothesisId: "H3",
        location: "AgentPanelView.jsx:265",
        message: "check flex model impact",
        data: {
          flexGrow: buttonStyle.flexGrow,
          flexShrink: buttonStyle.flexShrink,
          flexBasis: buttonStyle.flexBasis,
          parentClass: button.parentElement?.className || ""
        },
        timestamp: Date.now()
      })
    }).catch(() => {});
    // #endregion

    // #region agent log
    fetch("http://127.0.0.1:7934/ingest/2c4bcf68-efd6-4fd1-8130-1f5a368246bc", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "b84065" },
      body: JSON.stringify({
        sessionId: "b84065",
        runId: "baseline",
        hypothesisId: "H4",
        location: "AgentPanelView.jsx:266",
        message: "check icon intrinsic width",
        data: {
          iconClientWidth: icon?.clientWidth ?? null,
          iconFontSize: iconStyle?.fontSize ?? null,
          iconLineHeight: iconStyle?.lineHeight ?? null
        },
        timestamp: Date.now()
      })
    }).catch(() => {});
    // #endregion
  }, [active, messages.length]);

  return (
    <div className={`tab-panel console-module-panel ${active ? "active" : ""}`} id="panel-agent" aria-hidden={!active}>
      <div className="agent-layout">
        <div className="agent-sidebar">
          <h3 className="sidebar-title"><span className="material-icons">hub</span>连接与工具</h3>
          <div className="agent-settings-card md3-surface-container">
            <h4>外部 API 配置</h4>
            <div className="form-row stacked-row">
              <TextField label="Agent API 地址" value={apiUrl} onChange={setApiUrl} />
              <TextField label="API Token (可选)" value={apiKey} onChange={setApiKey} />
              <TextField label="模型名 / Endpoint ID" value={apiModel} onChange={setApiModel} />
            </div>
            <div className="launch-actions" style={{ marginTop: 12 }}>
              <Button variant="secondary" className="md-btn md-btn-tonal" onClick={saveConfig}>
                <span className="material-icons">save</span>保存配置
              </Button>
              <Button variant="outline" className="md-btn md-btn-outlined" onClick={testAgentApi}>
                <span className="material-icons">bolt</span>测试连接
              </Button>
            </div>
          </div>
          <div className="agent-common-tools md3-surface-container">
            <h4 className="tools-title"><span className="material-icons">build</span>常用工具</h4>
            <p className="tools-desc">点击后将对应问句发送到对话，并自动请求 Agent。</p>
            <div className="tools-actions" style={{ flexWrap: "wrap" }}>
              {AGENT_QUICK_PROMPTS.map((p) => (
                <Button
                  key={p.id}
                  type="button"
                  variant="outline"
                  className="md-btn md-btn-outlined md-btn-compact"
                  disabled={loading}
                  onClick={() => sendPresetMessage(p.text)}
                >
                  <span className="material-icons">chat</span>
                  {p.label}
                </Button>
              ))}
            </div>
          </div>
        </div>
        <div className="agent-main">
          <div className="agent-chat-panel">
            <div className="chat-header">
              <h3><span className="material-icons">chat</span>对话</h3>
              <div className="chat-controls">
                <Button
                  type="button"
                  size="icon"
                  variant="outline"
                  className="btn-icon-sm"
                  ref={exportButtonRef}
                  title="导出会话 JSON（审计）"
                  onClick={onExportSession}
                  aria-label="导出会话"
                >
                  <span ref={exportIconRef} className="material-icons">download</span>
                </Button>
                <Button size="icon" variant="outline" className="btn-icon-sm" onClick={onClearMessages}>
                  <span className="material-icons">delete_sweep</span>
                </Button>
              </div>
            </div>
            <div id="agent-chat-messages" ref={chatMessagesRef} className="chat-messages">
              {messages.map((msg, index) => (
                <div key={msg._messageId || `${msg.role}-${index}`} className={`chat-message ${msg.role}`}>
                  <div className={`message-avatar ${msg.role === "agent" ? "agent-avatar" : "user-avatar"}`}>
                    <span className="material-icons">{msg.role === "agent" ? "smart_toy" : "person"}</span>
                  </div>
                  <div className="message-content">
                    <ChatBubbleBody
                      role={msg.role}
                      content={msg.content}
                      modelName={msg.modelName}
                      reasoningApi={msg.reasoningApi}
                      toolsUsed={msg.toolsUsed}
                      streaming={msg.streaming}
                      messageKind={msg.kind}
                      traceRounds={msg.traceRounds}
                      traceOpen={msg.traceOpen}
                      bubbleIndex={index}
                      isLatestAgentBubble={msg.role === "agent" && index === latestAgentIndex}
                      relayMayMissReasoning={
                        msg.role === "agent" && relayReasoningHint && !msg.reasoningApi && !msg.kind
                      }
                    />
                  </div>
                </div>
              ))}
              {loading && !hasTailStreamingAgent ? (
                <div className="chat-message agent">
                  <div className="message-avatar agent-avatar">
                    <span className="material-icons">smart_toy</span>
                  </div>
                  <div className="message-content">
                    <div>处理中...</div>
                  </div>
                </div>
              ) : null}
            </div>
            {approvalOpen ? (
              <div
                className="agent-approval-in-chat agent-approval-sidebar-frame md3-surface-container"
                role="region"
                aria-labelledby="agent-approval-dialog-title"
              >
                <h2 id="agent-approval-dialog-title" className="md-dialog-title md3-dialog-headline">
                  批准修改训练配置？
                </h2>
                <p className="md-dialog-support md3-dialog-supporting">
                  确认后请使用下方按钮让 Agent 调用工具写入 configs/distill_config.yaml 并刷新训练配置表单。
                </p>
                {approvalChangeSummary && Array.isArray(approvalChangeSummary.paths) && approvalChangeSummary.paths.length > 0 ? (
                  <div className="agent-approval-diff-wrap">
                    <h3 className="agent-approval-diff-title">变更摘要（服务端核对）</h3>
                    <table className="agent-approval-diff-table">
                      <thead>
                        <tr>
                          <th scope="col">字段</th>
                          <th scope="col">类型</th>
                          <th scope="col">旧值</th>
                          <th scope="col">新值</th>
                        </tr>
                      </thead>
                      <tbody>
                        {approvalChangeSummary.paths.map((row) => (
                          <tr key={row.path}>
                            <td className="agent-diff-path">{row.path}</td>
                            <td>{row.kind || "—"}</td>
                            <td>
                              <code>{row.before === undefined ? "（缺失）" : JSON.stringify(row.before)}</code>
                            </td>
                            <td>
                              <code>{row.after === undefined ? "（缺失）" : JSON.stringify(row.after)}</code>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="agent-approval-diff-empty">
                    未检测到字段级差异（或与当前配置等价）；请结合上方聊天气泡中的摘要确认。
                  </p>
                )}
                <div className="md-dialog-actions md3-dialog-actions agent-approval-actions">
                  <Button type="button" variant="ghost" className="md-btn md-btn-text" onClick={onCloseApproval}>
                    取消
                  </Button>
                  <Button
                    type="button"
                    variant="default"
                    className="md-btn md-btn-filled primary md-btn-compact"
                    onClick={sendAgentExecuteApproval}
                    disabled={loading || !approvalToken}
                  >
                    <span className="material-icons">smart_toy</span>让 agent 执行
                  </Button>
                </div>
              </div>
            ) : null}
            <div className="chat-input-area">
              <div className="chat-input-wrapper">
                <div className={`md-field ${input ? "has-value" : ""}`}>
                  <textarea
                    ref={chatTextareaRef}
                    rows={1}
                    className="md-input"
                    value={input || ""}
                    onChange={(e) => {
                      setInput(e.target.value);
                      resizeChatInput();
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        send();
                      }
                    }}
                    placeholder=" "
                  />
                  <label className="md-field-label">输入消息</label>
                </div>
                <Button
                  className="md-btn md-btn-filled primary send-btn"
                  size="icon"
                  type="button"
                  onClick={send}
                  disabled={loading}
                >
                  <span className="material-icons">send</span>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AgentPanelView;
