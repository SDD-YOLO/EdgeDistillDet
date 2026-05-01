import { TrainingLauncher } from "./TrainingLauncher";
import { OutputConfigCard } from "./OutputConfigCard";
import { ResumeHistoryCard } from "./ResumeHistoryCard";
import { TrainingWeightsSection } from "./TrainingWeightsSection";
import { DistillationCoreParamsCard } from "./DistillationCoreParamsCard";
import { TrainingHyperparamsSection } from "./TrainingHyperparamsSection";
import { WandbConfigCard } from "./WandbConfigCard";
import { TrainingLogsPanel } from "./TrainingLogsPanel";

export function TrainingViewContainer({
  mode,
  running,
  onSwitchToDistillMode,
  onSwitchToResumeMode,
  onStartTraining,
  onStopTraining,
  onLoadConfigFromFile,
  onSaveConfig,
  onResetForm,
  isResumeStartDisabled,
  form,
  setNested,
  updateTrainingNested,
  applyComputePreset,
  isResumeConfigLocked,
  useDatasetApi,
  isRemoteApi,
  toast,
  pickLocalPath,
  logs,
  progress,
  progressPercent,
  autoScroll,
  setAutoScroll,
  clearLogs,
  downloadLogs,
  detectLogLevel,
  logContainerRef,
  resumeCandidates,
  selectedResumeIndex,
  setSelectedResumeIndex,
  onSelectResumeCandidate,
  currentOutputProject,
  renderedHint,
  runHint,
  isOutputPathOverlap,
  outputNameInputRef,
  pendingOverlapAlertRef,
  overlapAlertShownRef,
  refreshRunNameSuggestion,
  refreshResumeCandidates
}) {
  return (
    <>
      <section className="train-launcher">
        <TrainingLauncher
          mode={mode}
          running={running}
          onSwitchToDistillMode={onSwitchToDistillMode}
          onSwitchToResumeMode={onSwitchToResumeMode}
          onStartTraining={onStartTraining}
          onStopTraining={onStopTraining}
          onLoadConfigFromFile={onLoadConfigFromFile}
          onSaveConfig={onSaveConfig}
          onResetForm={onResetForm}
          isResumeStartDisabled={isResumeStartDisabled}
        />

        <div className="launcher-side">
          <OutputConfigCard
            form={form}
            isResumeMode={mode === "resume"}
            running={running}
            setNested={setNested}
            refreshRunNameSuggestion={refreshRunNameSuggestion}
            refreshResumeCandidates={refreshResumeCandidates}
            pickLocalPath={pickLocalPath}
            toast={toast}
            currentOutputProject={currentOutputProject}
            renderedHint={renderedHint}
            runHint={runHint}
            isOutputPathOverlap={isOutputPathOverlap}
            outputNameInputRef={outputNameInputRef}
            pendingOverlapAlertRef={pendingOverlapAlertRef}
            overlapAlertShownRef={overlapAlertShownRef}
          />
          <ResumeHistoryCard
            mode={mode}
            running={running}
            resumeCandidates={resumeCandidates}
            selectedResumeIndex={selectedResumeIndex}
            setSelectedResumeIndex={setSelectedResumeIndex}
            onSelectCandidate={onSelectResumeCandidate}
          />
        </div>
      </section>

      <div className="panel-grid">
        <>
          <div className="config-section">
            <TrainingWeightsSection form={form} setNested={setNested} pickLocalPath={pickLocalPath} toast={toast} running={running} isResumeConfigLocked={isResumeConfigLocked} />
            <DistillationCoreParamsCard form={form} setNested={setNested} running={running} isResumeConfigLocked={isResumeConfigLocked} />
            <TrainingHyperparamsSection
              form={form}
              setNested={setNested}
              updateTrainingNested={updateTrainingNested}
              applyComputePreset={applyComputePreset}
              running={running}
              isResumeConfigLocked={isResumeConfigLocked}
              useDatasetApi={useDatasetApi}
              isRemoteApi={isRemoteApi}
              toast={toast}
              pickLocalPath={pickLocalPath}
            />
            <WandbConfigCard form={form} setNested={setNested} running={running} />
          </div>
          <TrainingLogsPanel
            running={running}
            logs={logs}
            progress={progress}
            progressPercent={progressPercent}
            autoScroll={autoScroll}
            setAutoScroll={setAutoScroll}
            clearLogs={clearLogs}
            downloadLogs={downloadLogs}
            detectLogLevel={detectLogLevel}
            logContainerRef={logContainerRef}
          />
        </>
      </div>
    </>
  );
}