import { NumberField } from "../../../components/forms/NumberField";

export function DistillationCoreParamsCard({ form, setNested, running, isResumeConfigLocked }) {
  return (
    <div className="config-card config-card-compact">
      <h3 className="card-header">蒸馏核心参数</h3>
      <div className="form-grid training-form-grid">
        <NumberField label="初始 Alpha 权重" value={form.distillation.alpha_init} step="0.01" onChange={(v) => setNested("distillation", "alpha_init", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="温度上限 T_max" value={form.distillation.T_max} step="0.1" onChange={(v) => setNested("distillation", "T_max", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="温度下限 T_min" value={form.distillation.T_min} step="0.1" onChange={(v) => setNested("distillation", "T_min", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="训练预热轮数" value={form.distillation.warm_epochs} step="1" onChange={(v) => setNested("distillation", "warm_epochs", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="KD 损失权重 w_kd" value={form.distillation.w_kd} step="0.01" onChange={(v) => setNested("distillation", "w_kd", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="Focal KD 权重 w_focal" value={form.distillation.w_focal} step="0.01" onChange={(v) => setNested("distillation", "w_focal", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="特征对齐权重 w_feat" value={form.distillation.w_feat} step="0.01" onChange={(v) => setNested("distillation", "w_feat", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="小目标增强系数" value={form.distillation.scale_boost} step="0.1" onChange={(v) => setNested("distillation", "scale_boost", v)} disabled={running || isResumeConfigLocked} />
        <NumberField label="Focal Gamma 参数" value={form.distillation.focal_gamma} step="0.1" onChange={(v) => setNested("distillation", "focal_gamma", v)} disabled={running || isResumeConfigLocked} />
      </div>
    </div>
  );
}
