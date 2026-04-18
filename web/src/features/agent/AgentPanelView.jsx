import { TextField } from "../../components/forms/TextField";
import { AGENT_QUICK_PROMPTS, softenAgentBubbleText } from "./agentHelpers";

function ChatBubbleBody({ role, content, reasoningApi, toolsUsed, streaming, messageKind, traceRounds, relayMayMissReasoning }) {
  const rawText = content ?? "";
  const text =
    role === "agent" && messageKind !== "config_summary" ? softenAgentBubbleText(rawText, !!streaming) : rawText;
  const reasoningText = typeof reasoningApi === "string" && reasoningApi.trim() ? reasoningApi.trim() : "";
  const tools = Array.isArray(toolsUsed) ? toolsUsed.filter(Boolean) : [];
  const rounds = Array.isArray(traceRounds) ? traceRounds : [];
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
  const parts = [];
  const fenceRe = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let idx = 0;
  let m;
  while ((m = fenceRe.exec(text)) !== null) {
    if (m.index > idx) {
      parts.push({ kind: "text", value: text.slice(idx, m.index) });
    }
    parts.push({ kind: "code", value: (m[2] || "").replace(/\n$/, "") });
    idx = m.index + m[0].length;
  }
  if (idx < text.length) {
    parts.push({ kind: "text", value: text.slice(idx) });
  }
  if (parts.length === 0) {
    parts.push({ kind: "text", value: text });
  }
  const answerBlock = (
    <>
      {parts.map((p, i) =>
        p.kind === "code" ? (
          <pre key={i} className="agent-inline-codefence">
            {p.value}
          </pre>
        ) : (
          <div key={i} className="agent-inline-text">
            {p.value}
          </div>
        )
      )}
    </>
  );

  return (
    <div className="agent-bubble-formatted">
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
      {reasoningText ? (
        <details className="agent-reasoning-details" open={!!streaming}>
          <summary className="agent-reasoning-summary">
            思考过程
            {streaming ? <span className="agent-reasoning-live">生成中…</span> : null}
          </summary>
          <pre className="agent-reasoning-pre">{reasoningText}</pre>
        </details>
      ) : relayMayMissReasoning ? (
        <p className="agent-reasoning-missing">本连接未返回思考字段（非流式中继时部分服务商不提供）。</p>
      ) : null}
      {rounds.length > 0 ? (
        <details className="agent-trace-details">
          <summary className="agent-trace-summary">回合轨迹（可回溯）</summary>
          <ol className="agent-trace-list">
            {rounds.map((r, tidx) => (
              <li key={`trace-${tidx}-${r.round}`} className="agent-trace-item">
                <div className="agent-trace-round-title">第 {r.round} 轮</div>
                {r.reasoning ? <pre className="agent-trace-pre">{r.reasoning}</pre> : null}
                <div className="agent-trace-reply-snippet">{typeof r.reply === "string" ? r.reply.slice(0, 2000) : ""}</div>
                <div className="agent-trace-tool">
                  {r.tool?.name ? (
                    <>
                      <span className="agent-trace-tool-label">工具: {r.tool.name}</span>
                      <pre className="agent-trace-pre">{JSON.stringify(r.toolResultSummary ?? {}, null, 2)}</pre>
                    </>
                  ) : (
                    <span className="agent-trace-tool-none">本回合未通过 JSON 触发工具</span>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </details>
      ) : null}
      {reasoningText ? <div className="agent-answer-body">{answerBlock}</div> : answerBlock}
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
              <button className="md-btn md-btn-tonal" onClick={saveConfig}>
                <span className="material-icons">save</span>保存配置
              </button>
              <button className="md-btn md-btn-outlined" onClick={testAgentApi}>
                <span className="material-icons">bolt</span>测试连接
              </button>
            </div>
          </div>
          <div className="agent-common-tools md3-surface-container">
            <h4 className="tools-title"><span className="material-icons">build</span>常用工具</h4>
            <p className="tools-desc">点击后将对应问句发送到对话，并自动请求 Agent。</p>
            <div className="tools-actions" style={{ flexWrap: "wrap" }}>
              {AGENT_QUICK_PROMPTS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className="md-btn md-btn-outlined md-btn-compact"
                  disabled={loading}
                  onClick={() => sendPresetMessage(p.text)}
                >
                  <span className="material-icons">chat</span>
                  {p.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="agent-main">
          <div className="agent-chat-panel">
            <div className="chat-header">
              <h3><span className="material-icons">chat</span>对话</h3>
              <div className="chat-controls">
                <button
                  type="button"
                  className="btn-icon-sm"
                  title="导出会话 JSON（审计）"
                  onClick={onExportSession}
                  aria-label="导出会话"
                >
                  <span className="material-icons">download</span>
                </button>
                <button className="btn-icon-sm" onClick={onClearMessages}><span className="material-icons">delete_sweep</span></button>
              </div>
            </div>
            <div id="agent-chat-messages" ref={chatMessagesRef} className="chat-messages">
              {messages.map((msg, index) => (
                <div key={`${msg.role}-${index}`} className={`chat-message ${msg.role}`}>
                  <div className={`message-avatar ${msg.role === "agent" ? "agent-avatar" : "user-avatar"}`}>
                    <span className="material-icons">{msg.role === "agent" ? "smart_toy" : "person"}</span>
                  </div>
                  <div className="message-content">
                    <ChatBubbleBody
                      role={msg.role}
                      content={msg.content}
                      reasoningApi={msg.reasoningApi}
                      toolsUsed={msg.toolsUsed}
                      streaming={msg.streaming}
                      messageKind={msg.kind}
                      traceRounds={msg.traceRounds}
                      relayMayMissReasoning={
                        msg.role === "agent" && relayReasoningHint && !msg.reasoningApi && !msg.kind
                      }
                    />
                  </div>
                </div>
              ))}
              {loading && !messages.some((m) => m.streaming) ? (
                <div className="chat-message agent">
                  <div className="message-avatar agent-avatar"><span className="material-icons">smart_toy</span></div>
                  <div className="message-content"><div>处理中...</div></div>
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
                            <td><code>{row.before === undefined || row.before === null ? "—" : JSON.stringify(row.before)}</code></td>
                            <td><code>{row.after === undefined || row.after === null ? "—" : JSON.stringify(row.after)}</code></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="agent-approval-diff-empty">未检测到字段级差异（或与当前配置等价）；请结合上方聊天气泡中的摘要确认。</p>
                )}
                <div className="md-dialog-actions md3-dialog-actions agent-approval-actions">
                  <button type="button" className="md-btn md-btn-text" onClick={onCloseApproval}>
                    取消
                  </button>
                  <button
                    type="button"
                    className="md-btn md-btn-filled primary md-btn-compact"
                    onClick={sendAgentExecuteApproval}
                    disabled={loading || !approvalToken}
                  >
                    <span className="material-icons">smart_toy</span>让 agent 执行
                  </button>
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
                  >
                  </textarea>
                  <label className="md-field-label">输入消息</label>
                </div>
                <button className="md-btn md-btn-filled primary send-btn" type="button" onClick={send} disabled={loading}>
                  <span className="material-icons">send</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AgentPanelView;
