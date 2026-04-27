import { SelectField } from "../../../components/forms/SelectField";
import { TextField } from "../../../components/forms/TextField";

export function WandbConfigCard({ form, setNested, running }) {
  return (
    <div className="config-card config-card-compact">
      <h3 className="card-header">W&B 配置</h3>
      <div className="form-grid training-form-grid">
        <div className="form-group switch-group">
          <label>启用 Weights & Biases</label>
          <label className="md-switch">
            <input
              type="checkbox"
              checked={Boolean(form.wandb?.enabled)}
              onChange={(e) => setNested("wandb", "enabled", e.target.checked)}
              disabled={running}
            />
          </label>
        </div>
        <SelectField
          label="运行模式"
          value={form.wandb?.mode || "online"}
          onChange={(v) => setNested("wandb", "mode", v)}
          options={[
            { value: "online", label: "online" },
            { value: "offline", label: "offline" },
            { value: "disabled", label: "disabled" }
          ]}
          disabled={running || !form.wandb?.enabled}
        />
        <TextField
          label="Project"
          value={form.wandb?.project || ""}
          onChange={(v) => setNested("wandb", "project", v)}
          disabled={running || !form.wandb?.enabled}
        />
        <TextField
          label="Entity (可选)"
          value={form.wandb?.entity || ""}
          onChange={(v) => setNested("wandb", "entity", v)}
          disabled={running || !form.wandb?.enabled}
        />
        <TextField
          label="Run Name (可选)"
          value={form.wandb?.name || ""}
          onChange={(v) => setNested("wandb", "name", v)}
          disabled={running || !form.wandb?.enabled}
        />
        <TextField
          label="Group (可选)"
          value={form.wandb?.group || ""}
          onChange={(v) => setNested("wandb", "group", v)}
          disabled={running || !form.wandb?.enabled}
        />
        <TextField
          label="Job Type (可选)"
          value={form.wandb?.job_type || ""}
          onChange={(v) => setNested("wandb", "job_type", v)}
          disabled={running || !form.wandb?.enabled}
        />
        <TextField
          label="Tags (逗号分隔，可选)"
          value={form.wandb?.tags || ""}
          onChange={(v) => setNested("wandb", "tags", v)}
          disabled={running || !form.wandb?.enabled}
        />
        <TextField
          label="Notes (可选)"
          value={form.wandb?.notes || ""}
          onChange={(v) => setNested("wandb", "notes", v)}
          disabled={running || !form.wandb?.enabled}
        />
      </div>
    </div>
  );
}
