import { PathField } from "../../../components/forms/PathField";

export function OutputConfigCard({
  form,
  isResumeMode,
  running,
  setNested,
  refreshRunNameSuggestion,
  refreshResumeCandidates,
  pickLocalPath,
  toast,
  currentOutputProject,
  renderedHint,
  runHint,
  isOutputPathOverlap,
  outputNameInputRef,
  pendingOverlapAlertRef,
  overlapAlertShownRef
}) {
  return (
    <div className="config-card launcher-side-card">
      <h3 className="card-header">输出配置</h3>
      <div className="form-row stacked-row">
        <div className="form-group">
          <PathField
            label="项目目录"
            value={form.output.project || ""}
            onChange={(project) => {
              if (isResumeMode) return;
              setNested("output", "project", project);
              refreshRunNameSuggestion(project || "runs", form.output.name, true);
              refreshResumeCandidates(project || "runs", false);
            }}
            onBrowse={async () => {
              if (isResumeMode) return;
              try {
                const selected = await pickLocalPath({
                  kind: "directory",
                  title: "选择训练输出项目目录",
                  initialPath: form.output.project || "runs"
                });
                if (!selected) return;
                setNested("output", "project", selected);
                refreshRunNameSuggestion(selected || "runs", form.output.name, true);
                refreshResumeCandidates(selected || "runs", false);
              } catch (error) {
                toast(error.message, "error");
              }
            }}
            disabled={running || isResumeMode}
          />
        </div>
        <div className="form-group">
          <div className={`md-field ${(form.output.name || "").trim() ? "has-value" : ""}`}>
            <input
              ref={outputNameInputRef}
              className="md-input"
              placeholder=" "
              value={form.output.name || ""}
              onChange={(e) => {
                const nextName = e.target.value;
                setNested("output", "name", nextName);
              }}
              onBlur={() => {
                if (isResumeMode) return;
                const overlapKey = `${currentOutputProject}/${(form.output.name || "").trim()}`;
                const shouldAlertOnBlur = Boolean(
                  isOutputPathOverlap &&
                  (pendingOverlapAlertRef.current || overlapAlertShownRef.current !== overlapKey)
                );
                if (shouldAlertOnBlur) {
                  pendingOverlapAlertRef.current = false;
                  overlapAlertShownRef.current = overlapKey;
                  window.alert(`路径重合：${overlapKey}`);
                  toast(`路径重合：${overlapKey}`, "warning");
                }
              }}
              disabled={running || isResumeMode}
            />
            <label className="md-field-label">运行名称</label>
          </div>
          <small className={`hint ${isOutputPathOverlap ? "warning" : ""}`}>{renderedHint || runHint}</small>
        </div>
      </div>
    </div>
  );
}
