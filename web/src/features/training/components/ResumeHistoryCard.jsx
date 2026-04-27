import { M3Select } from "../../../components/forms/M3Select";

export function ResumeHistoryCard({
  mode,
  running,
  resumeCandidates,
  selectedResumeIndex,
  setSelectedResumeIndex,
  onSelectCandidate
}) {
  return (
    <div className={`config-card launcher-side-card ${mode !== "resume" ? "disabled-panel" : ""}`}>
      <h3 className="card-header">续训历史</h3>
      <div className="form-group">
        <label>请选择历史运行</label>
        <M3Select
          value={String(selectedResumeIndex)}
          onChange={(nextValue) => {
            const idx = Number(nextValue) || 0;
            setSelectedResumeIndex(idx);
            onSelectCandidate(idx);
          }}
          options={
            resumeCandidates.length === 0
              ? [{ value: "0", label: "暂无可用候选" }]
              : resumeCandidates.map((item, idx) => ({ value: String(idx), label: item.display_name }))
          }
          disabled={mode !== "resume" || running || resumeCandidates.length === 0}
          ariaLabel="请选择历史运行"
        />
      </div>
    </div>
  );
}
