import { NumberField } from "../../../components/forms/NumberField";
import { PathField } from "../../../components/forms/PathField";
import { SelectField } from "../../../components/forms/SelectField";
import { TextField } from "../../../components/forms/TextField";

export function TrainingHyperparamsSection({
  form,
  setNested,
  updateTrainingNested,
  applyComputePreset,
  running,
  isResumeConfigLocked,
  useDatasetApi,
  isRemoteApi,
  toast,
  pickLocalPath
}) {
  return (
    <div className="config-card config-card-compact">
      <h3 className="card-header">训练超参数</h3>
      <div className="form-grid training-form-grid">
        <PathField
          label="数据集配置文件"
          value={form.training.data_yaml}
          onChange={(v) => setNested("training", "data_yaml", v)}
          onBrowse={async () => {
            try {
              const selected = await pickLocalPath({
                kind: "file",
                title: "选择数据集配置文件",
                initialPath: form.training.data_yaml || "",
                filters: [
                  { name: "YAML Files", patterns: ["*.yaml", "*.yml"] },
                  { name: "All Files", patterns: ["*.*"] }
                ]
              });
              if (selected) setNested("training", "data_yaml", selected);
            } catch (error) {
              toast(error.message, "error");
            }
          }}
          disabled={running || useDatasetApi || isResumeConfigLocked}
        />

        <SelectField
          label="云算力配置"
          value={form.training.compute_provider || "local"}
          onChange={(v) => applyComputePreset(v)}
          options={[
            { value: "local", label: "本地" },
            { value: "autodl", label: "autoDL 云算力" },
            { value: "colab", label: "Google Colab 云算力" },
            { value: "remote_api", label: "远程云算力 API" }
          ]}
          disabled={running || isResumeConfigLocked}
        />

        {isRemoteApi ? (
          <>
            <SelectField
              label="数据集来源"
              value={form.training?.dataset_api?.source || (form.training?.dataset_api?.enabled ? "api" : "path")}
              onChange={(v) => updateTrainingNested("dataset_api", { source: v, enabled: v === "api" })}
              options={[
                { value: "path", label: "本地/YAML 路径" },
                { value: "api", label: "数据集 API" }
              ]}
              disabled={running || isResumeConfigLocked}
            />
            <TextField
              label="云训练 API Base URL"
              value={form.training?.cloud_api?.base_url || ""}
              onChange={(v) => updateTrainingNested("cloud_api", { base_url: v })}
              disabled={running || isResumeConfigLocked}
            />
            <TextField
              label="云训练 API Token (可选)"
              value={form.training?.cloud_api?.token || ""}
              onChange={(v) => updateTrainingNested("cloud_api", { token: v })}
              disabled={running || isResumeConfigLocked}
            />
            {useDatasetApi ? (
              <>
                <TextField
                  label="数据集 API URL"
                  value={form.training?.dataset_api?.resolve_url || ""}
                  onChange={(v) => updateTrainingNested("dataset_api", { resolve_url: v })}
                  disabled={running || isResumeConfigLocked}
                />
                <TextField
                  label="数据集 API Token (可选，默认复用云训练 Token)"
                  value={form.training?.dataset_api?.token || ""}
                  onChange={(v) => updateTrainingNested("dataset_api", { token: v })}
                  disabled={running || isResumeConfigLocked}
                />
                <TextField
                  label="数据集名称/别名 (可选)"
                  value={form.training?.dataset_api?.dataset_name || ""}
                  onChange={(v) => updateTrainingNested("dataset_api", { dataset_name: v })}
                  disabled={running || isResumeConfigLocked}
                />
              </>
            ) : null}
          </>
        ) : null}

        {form.training.compute_provider === "local" ? (
          <SelectField
            label="设备"
            value={form.training.device}
            onChange={(v) => setNested("training", "device", v)}
            options={[
              { value: "0", label: "GPU 0" },
              { value: "1", label: "GPU 1" },
              { value: "cpu", label: "CPU" }
            ]}
            disabled={running || isResumeConfigLocked}
          />
        ) : null}

        <NumberField label="训练轮数 Epochs" value={form.training.epochs} step="1" onChange={(v) => setNested("training", "epochs", v)} disabled={running || isResumeConfigLocked} />
        <SelectField
          label="图像尺寸 imgsz"
          value={form.training.imgsz}
          onChange={(v) => setNested("training", "imgsz", Number(v))}
          options={[
            { value: "320", label: "320" },
            { value: "416", label: "416" },
            { value: "512", label: "512" },
            { value: "640", label: "640" },
            { value: "768", label: "768" },
            { value: "960", label: "960" },
            { value: "1024", label: "1024" },
            { value: "1280", label: "1280" }
          ]}
          disabled={running || isResumeConfigLocked}
        />
        <NumberField label="Batch Size" value={form.training.batch} step="1" onChange={(v) => setNested("training", "batch", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="初始学习率 lr0" value={form.training.lr0} step="0.001" onChange={(v) => setNested("training", "lr0", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="最终学习率因子 lrf" value={form.training.lrf} step="0.01" onChange={(v) => setNested("training", "lrf", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="学习率预热轮数" value={form.training.warmup_epochs} step="0.5" onChange={(v) => setNested("training", "warmup_epochs", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="Mosaic 增强概率" value={form.training.mosaic} step="0.01" onChange={(v) => setNested("training", "mosaic", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="Mixup 增强概率" value={form.training.mixup} step="0.01" onChange={(v) => setNested("training", "mixup", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="关闭 Mosaic 的 epoch" value={form.training.close_mosaic} step="1" onChange={(v) => setNested("training", "close_mosaic", v)} disabled={running || isResumeConfigLocked} />
        <div className="form-group switch-group">
          <label>AMP 混合精度训练</label>
          <label className="md-switch">
            <input type="checkbox" checked={Boolean(form.training.amp)} onChange={(e) => setNested("training", "amp", e.target.checked)} disabled={running || isResumeConfigLocked} />
          </label>
        </div>
      </div>
    </div>
  );
}
