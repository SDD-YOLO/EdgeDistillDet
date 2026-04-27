import { Button } from "../../../components/ui/button";
import { TrainingModeSelector } from "./TrainingModeSelector";

export function TrainingLauncher({
  mode,
  running,
  onSwitchToDistillMode,
  onSwitchToResumeMode,
  onStartTraining,
  onStopTraining,
  onLoadConfigFromFile,
  onSaveConfig,
  onResetForm,
  isResumeStartDisabled
}) {
  return (
    <div className="launcher-left">
      <div className="launch-info">
        <div className="launch-header">
          <h2><span className="material-icons">rocket_launch</span> 训练控制台</h2>
          <span className={`badge ${running ? "running" : "idle"}`}>{running ? "训练中" : "就绪"}</span>
        </div>
        <p className="launch-desc">配置参数后选择训练模式并启动训练任务</p>
      </div>

      <TrainingModeSelector
        mode={mode}
        running={running}
        onSwitchToDistillMode={onSwitchToDistillMode}
        onSwitchToResumeMode={onSwitchToResumeMode}
      />

      <div className="launch-actions">
        <Button
          id="btn-start-training"
          className="btn-start"
          variant="default"
          onClick={onStartTraining}
          disabled={running || isResumeStartDisabled}
          title={isResumeStartDisabled ? "当前没有可用断点，无法开始断点续训" : ""}
        >
          <span className="material-icons">play_arrow</span>开始训练
        </Button>
        <Button id="btn-stop-training" className="btn-stop" variant="destructive" onClick={onStopTraining} disabled={!running}>
          <span className="material-icons">stop</span>停止训练
        </Button>
        <label className="md-btn md-btn-tonal cursor-pointer">
          <span className="material-icons">file_open</span>加载配置
          <input type="file" accept=".yaml,.yml" style={{ display: "none" }} onChange={onLoadConfigFromFile} />
        </label>
        <Button variant="outline" onClick={onSaveConfig}>
          <span className="material-icons">save</span>保存配置
        </Button>
        <Button variant="ghost" onClick={onResetForm}>
          <span className="material-icons">refresh</span>重置表单
        </Button>
      </div>
    </div>
  );
}
