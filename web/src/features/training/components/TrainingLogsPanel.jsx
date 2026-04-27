import { Button } from "../../../components/ui/button";

export function TrainingLogsPanel({
  running,
  logs,
  progress,
  progressPercent,
  autoScroll,
  setAutoScroll,
  clearLogs,
  downloadLogs,
  detectLogLevel,
  logContainerRef
}) {
  return (
    <div className="log-section">
      <div className="log-card">
        <div className="log-header">
          <h3><span className="material-icons">terminal</span>训练日志</h3>
          <div className="log-controls">
            <span className={`badge ${running ? "running" : "idle"}`}>{running ? "训练中" : "空闲"}</span>
            <Button size="icon" variant="outline" className="btn-icon-sm" onClick={clearLogs} title="清空日志"><span className="material-icons">delete_outline</span></Button>
            <Button size="icon" variant="outline" className="btn-icon-sm" onClick={downloadLogs} title="下载日志"><span className="material-icons">download</span></Button>
            <Button
              size="icon"
              variant="outline"
              className={`btn-icon-sm ${!autoScroll ? "is-disabled" : ""}`}
              onClick={() => setAutoScroll((prev) => !prev)}
              title="自动滚动"
            >
              <span className="material-icons">vertical_align_bottom</span>
            </Button>
          </div>
        </div>

        <div className="progress-container" style={{ display: running ? "block" : "none" }}>
          <div className="progress-bar-wrapper">
            <div className="progress-bar" style={{ width: `${progressPercent.toFixed(1)}%` }} />
          </div>
          <div className="progress-info">
            <span>{`Epoch: ${progress.current} / ${progress.total || "-"}`}</span>
            <span>{`耗时: ${progress.elapsed}`}</span>
            <span>{`预计总耗时: ${progress.expected}`}</span>
          </div>
        </div>

        <div ref={logContainerRef} className="log-container">
          {logs.length === 0 ? <div className="log-line info log-empty-placeholder">暂无日志输出</div> : null}
          {logs.map((line, index) => {
            const level = detectLogLevel(line);
            return (
              <div key={`${index}-${String(line).slice(0, 18)}`} className={`log-line ${level}`}>
                {String(line)}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
