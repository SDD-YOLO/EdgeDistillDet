import { Button } from "../../../components/ui/button";
import { PathField } from "../../../components/forms/PathField";
import { SelectField } from "../../../components/forms/SelectField";
import { detectLogLevel } from "../../../utils/logging";

export function ExportLauncherPanel({
  exportRunning,
  exportReady,
  startExport,
  stopExport,
  exportAutoScroll,
  setExportAutoScroll,
  clearExportLogs,
  downloadExportLogs,
  exportLogs,
  exportLogContainerRef,
  exportSectionCards,
  renderAdvancedField,
  exportWeight,
  onExportWeightChange,
  onExportWeightBrowse,
  exportWeightCandidates,
  selectedExportWeightIndex,
  onSelectExportWeightCandidate,
}) {
  return (
    <section className="train-launcher launcher-column">
      <div className="launcher-left">
        <div className="launch-info">
          <div className="launch-header">
            <h2><span className="material-icons">save_alt</span> 导出控制台</h2>
            <span className={`badge ${exportRunning ? "running" : "idle"}`}>{exportRunning ? "导出中" : "空闲"}</span>
          </div>
          <p className="launch-desc">通过该控制台启动或停止模型导出，并查看终端日志。</p>
        </div>
        <div className="launch-actions">
          <Button variant="default" onClick={startExport} disabled={!exportReady || exportRunning}>
            <span className="material-icons">play_arrow</span> 开始导出
          </Button>
          <Button variant="destructive" onClick={stopExport} disabled={!exportRunning}>
            <span className="material-icons">stop</span> 停止导出
          </Button>
        </div>
      </div>
      <div className="launcher-side">
        <div className="config-card launcher-side-card">
          <h3 className="card-header">导出终端日志</h3>
          <div className="card-body">
            <div className="log-header">
              <div className="log-controls">
                <Button size="icon" variant="outline" className="btn-icon-sm" onClick={clearExportLogs} title="清空日志">
                  <span className="material-icons">delete_outline</span>
                </Button>
                <Button size="icon" variant="outline" className="btn-icon-sm" onClick={downloadExportLogs} title="下载日志">
                  <span className="material-icons">download</span>
                </Button>
                <Button
                  size="icon"
                  variant="outline"
                  className={`btn-icon-sm ${!exportAutoScroll ? "is-disabled" : ""}`}
                  onClick={() => setExportAutoScroll((prev) => !prev)}
                  title="自动滚动"
                >
                  <span className="material-icons">vertical_align_bottom</span>
                </Button>
              </div>
            </div>
            <div ref={exportLogContainerRef} className="log-container">
              {exportLogs.length === 0 ? (
                <div className="log-line info log-empty-placeholder">暂无导出日志输出</div>
              ) : null}
              {exportLogs.map((line, index) => {
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
        <div className="config-card launcher-side-card">
          <h3 className="card-header">导出参数</h3>
          <div className="card-body">
            <div className="form-grid training-form-grid">
              <div className="config-card config-card-compact">
                <div className="form-grid training-form-grid">
                  <div>
                    <PathField
                      label="权重文件"
                      value={exportWeight || ""}
                      onChange={onExportWeightChange}
                      onBrowse={onExportWeightBrowse}
                      disabled={exportRunning}
                    />
                  </div>
                  <div>
                    <SelectField
                      label="训练权重候选"
                      value={String(selectedExportWeightIndex)}
                      onChange={onSelectExportWeightCandidate}
                      options={
                        exportWeightCandidates.length === 0
                          ? [{ value: "0", label: "暂无可用权重" }]
                          : exportWeightCandidates.map((item, idx) => ({
                              value: String(idx),
                              label: item.display_name,
                            }))
                      }
                      disabled={exportRunning || exportWeightCandidates.length === 0}
                    />
                  </div>
                </div>
              </div>
            </div>
            <div className="form-grid training-form-grid">
              {exportSectionCards.map(({ scope, section }) => (
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
      </div>
    </section>
  );
}
