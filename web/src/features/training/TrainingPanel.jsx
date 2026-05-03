import { CONFIG_GROUPS } from "../../constants/configGroups";
import UnifiedConfigPanel from "../config-center/UnifiedConfigPanel";
import { useTrainingPanelController } from "./hooks/useTrainingPanelController";

function TrainingPanel({ toast, active, view = "training" }) {
  const controller = useTrainingPanelController({ toast });

  return (
    <div
      className={`tab-panel console-module-panel ${active ? "active" : ""}`}
      id="panel-training"
      aria-hidden={!active}
    >
      <UnifiedConfigPanel
        groups={CONFIG_GROUPS}
        form={controller.form}
        getValue={controller.getValueByPath}
        setValue={controller.setValueByPath}
        pickLocalPath={controller.pickLocalPath}
        previewPayload={controller.previewPayload}
        running={controller.running}
        isResumeMode={false}
        renderedHint={controller.renderedHint}
        runHint={controller.runHint}
        onSave={() =>
          controller
            .saveConfig()
            .then(() => toast("配置已保存", "success"))
            .catch((e) => toast(e.message, "error"))
        }
        onLoad={controller.loadConfigFromFile}
        onReset={controller.resetForm}
        onStartTraining={controller.startTraining}
        onStopTraining={controller.stopTraining}
        onStartDisplay={controller.startInference}
        onStopDisplay={controller.stopInference}
        onStartExport={controller.startExport}
        onStopExport={controller.stopExport}
        trainingRunning={controller.running}
        displayRunning={controller.inferRunning}
        exportRunning={controller.exportRunning}
      />
    </div>
  );
}
export default TrainingPanel;
