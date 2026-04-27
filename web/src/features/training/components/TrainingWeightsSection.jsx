import { NumberField } from "../../../components/forms/NumberField";
import { PathField } from "../../../components/forms/PathField";

export function TrainingWeightsSection({ form, setNested, pickLocalPath, toast, running, isResumeConfigLocked }) {
  return (
    <div className="config-card config-card-compact">
      <h3 className="card-header">模型权重</h3>
      <div className="form-row">
        <div className="form-group flex-2">
          <PathField
            label="学生模型权重"
            value={form.distillation.student_weight || ""}
            onChange={(v) => setNested("distillation", "student_weight", v)}
            onBrowse={async () => {
              try {
                const selected = await pickLocalPath({
                  kind: "file",
                  title: "选择学生模型权重文件",
                  initialPath: form.distillation.student_weight || "",
                  filters: [
                    { name: "PyTorch Weights", patterns: ["*.pt", "*.pth"] },
                    { name: "All Files", patterns: ["*.*"] }
                  ]
                });
                if (selected) setNested("distillation", "student_weight", selected);
              } catch (error) {
                toast(error.message, "error");
              }
            }}
            disabled={running || isResumeConfigLocked}
          />
        </div>
        <div className="form-group flex-2">
          <PathField
            label="教师模型权重"
            value={form.distillation.teacher_weight || ""}
            onChange={(v) => setNested("distillation", "teacher_weight", v)}
            onBrowse={async () => {
              try {
                const selected = await pickLocalPath({
                  kind: "file",
                  title: "选择教师模型权重文件",
                  initialPath: form.distillation.teacher_weight || "",
                  filters: [
                    { name: "PyTorch Weights", patterns: ["*.pt", "*.pth"] },
                    { name: "All Files", patterns: ["*.*"] }
                  ]
                });
                if (selected) setNested("distillation", "teacher_weight", selected);
              } catch (error) {
                toast(error.message, "error");
              }
            }}
            disabled={running || isResumeConfigLocked}
          />
        </div>
      </div>
    </div>
  );
}
