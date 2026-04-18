import { TextField } from "../../components/forms/TextField";
import { AGENT_QUICK_PROMPTS } from "./agentHelpers";

function ChatBubbleBody({ role, content, reasoningApi, toolsUsed, streaming }) {
  const text = content ?? "";
  const reasoningText = typeof reasoningApi === "string" && reasoningApi.trim() ? reasoningApi.trim() : "";
  const tools = Array.isArray(toolsUsed) ? toolsUsed.filter(Boolean) : [];
  if (role !== "agent") {
    return <div className="chat-plain chat-pre-wrap">{text}</div>;
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
  loadSchema,
  parseClipboardPatch,
  generateMetricsReport,
  approvalOpen,
  approvalBody,
  onCloseApproval,
  sendAgentExecuteApproval,
  approvalToken,
  sendPresetMessage,
  messages,
  onClearMessages,
  chatMessagesRef,
  chatTextareaRef,
  input,
  setInput,
  resizeChatInput,
  send,
  outputText,
  outputIsPlaceholder
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
            <div className="agent-sidebar-aux-tools">
              <button type="button" className="md-btn md-btn-tonal md-btn-compact" onClick={loadSchema} disabled={loading}>
                <span className="material-icons">schema</span>配置结构
              </button>
              <button type="button" className="md-btn md-btn-outlined md-btn-compact" onClick={parseClipboardPatch} disabled={loading}>
                <span className="material-icons">content_paste</span>剪贴板 Patch
              </button>
              <button type="button" className="md-btn md-btn-tonal md-btn-compact" onClick={generateMetricsReport} disabled={loading}>
                <span className="material-icons">summarize</span>指标快照
              </button>
            </div>
          </div>
          <div
            className="agent-approval-sidebar agent-approval-sidebar-frame md3-surface-container"
            role="region"
            aria-labelledby="agent-approval-dialog-title"
          >
            {approvalOpen ? (
              <>
                <h2 id="agent-approval-dialog-title" className="md-dialog-title md3-dialog-headline">
                  批准修改训练配置？
                </h2>
                <p className="md-dialog-support md3-dialog-supporting">
                  确认后请使用下方按钮让 Agent 调用工具写入 configs/distill_config.yaml 并刷新训练配置表单。
                </p>
                <pre className="md-dialog-pre md3-dialog-body-scroll">{approvalBody}</pre>
                <div className="md-dialog-actions md3-dialog-actions agent-approval-sidebar-actions">
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
              </>
            ) : (
              <>
                <h2 id="agent-approval-dialog-title" className="md-dialog-title agent-approval-sidebar-idle-title">
                  <span className="material-icons" aria-hidden>
                    verified_user
                  </span>
                  审批区
                </h2>
                <p className="tools-desc agent-approval-sidebar-idle-desc">
                  外部 Agent 返回的 patch 仅预览；通过预览签发票据后，将在此显示 YAML 与 merged_preview，并可使用「让 agent 执行」写入配置。
                </p>
              </>
            )}
          </div>
          <div className="agent-common-tools md3-surface-container">
            <h4 className="tools-title"><span className="material-icons">build</span>常用工具</h4>
            <p className="tools-desc">点击后将对应问句发送到左侧对话，并自动请求 Agent；右侧输出优先展示可复制的终端命令。</p>
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
          <div className="agent-status-panel log-card">
            <div className="status-header log-header">
              <h3><span className="material-icons">terminal</span>Agent 输出</h3>
            </div>
            <div id="agent-output" className="agent-output log-container">
              <pre id="agent-output-content" className={outputIsPlaceholder ? "agent-output-placeholder" : ""}>{outputText}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AgentPanelView;
