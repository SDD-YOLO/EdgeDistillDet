export function TrainingModeSelector({
  mode,
  running,
  onSwitchToDistillMode,
  onSwitchToResumeMode
}) {
  return (
    <div className="launch-modes">
      <label className={`mode-option ${mode === "distill" ? "selected" : ""}`}>
        <input type="radio" checked={mode === "distill"} onChange={onSwitchToDistillMode} disabled={running} />
        <div className="mode-card">
          <span className="material-icons mode-icon">school</span>
          <div className="mode-text">
            <strong>蒸馏训练</strong>
            <span>知识蒸馏训练，训练完成后自动评估模型性能</span>
          </div>
        </div>
      </label>
      <label className={`mode-option ${mode === "resume" ? "selected" : ""}`}>
        <input type="radio" checked={mode === "resume"} onChange={onSwitchToResumeMode} disabled={running} />
        <div className="mode-card">
          <span className="material-icons mode-icon">restart_alt</span>
          <div className="mode-text">
            <strong>断点续训</strong>
            <span>从上次检查点恢复训练进度</span>
          </div>
        </div>
      </label>
    </div>
  );
}
