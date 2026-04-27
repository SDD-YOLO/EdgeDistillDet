import { Button } from "../../../components/ui/button";

export function DisplayLauncherPanel({
  inferRunning,
  startInference,
  stopInference,
  displaySectionCards,
  renderAdvancedField
}) {
  return (
    <section className="train-launcher launcher-column">
      <div className="launcher-left">
        <div className="launch-info">
          <div className="launch-header">
            <h2><span className="material-icons">visibility</span> 推理控制台</h2>
            <span className={`badge ${inferRunning ? "running" : "idle"}`}>{inferRunning ? "推理中" : "空闲"}</span>
          </div>
          <p className="launch-desc">通过该控制台启动或停止推理，并在右侧查看结果预览。</p>
        </div>
        <div className="launch-actions">
          <Button variant="default" className="btn-start" onClick={startInference} disabled={inferRunning}>
            <span className="material-icons">play_arrow</span> 开始推理
          </Button>
          <Button variant="destructive" className="btn-stop" onClick={stopInference} disabled={!inferRunning}>
            <span className="material-icons">stop</span> 停止推理
          </Button>
        </div>
      </div>
      <div className="launcher-side">
        <div className="config-card launcher-side-card">
          <h3 className="card-header">推理结果预览</h3>
          <div className="card-body display-preview-container">
            <div className="display-preview-placeholder">推理结果将在此处显示</div>
          </div>
        </div>
        <div className="config-card launcher-side-card">
          <h3 className="card-header">推理参数</h3>
          <div className="card-body">
            {displaySectionCards.map(({ scope, section }) => (
              <div key={`${scope}-${section.title}`} className="config-card config-card-compact">
                <h3 className="card-header">{section.title}</h3>
                <div className="form-grid training-form-grid">
                  {(section.params || []).map((param) => (
                    <div key={param.key}>{renderAdvancedField(scope, param)}</div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
