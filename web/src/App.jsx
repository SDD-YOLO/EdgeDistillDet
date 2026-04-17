import { useEffect, useRef, useState } from "react";
import {
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  Legend,
  LineController,
  LineElement,
  LinearScale,
  LogarithmicScale,
  PointElement,
  Tooltip
} from "chart.js";

Chart.register(
  BarController,
  BarElement,
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  LineController,
  Legend,
  Tooltip
);

const DEFAULT_FORM = {
  distillation: {
    student_weight: "",
    teacher_weight: "",
    alpha_init: 0.5,
    T_max: 6,
    T_min: 1.5,
    warm_epochs: 5,
    w_kd: 0.5,
    w_focal: 0.3,
    w_feat: 0,
    scale_boost: 2,
    focal_gamma: 2
  },
  training: {
    data_yaml: "",
    device: "0",
    epochs: 150,
    imgsz: 640,
    batch: -1,
    workers: 0,
    lr0: 0.01,
    lrf: 0.1,
    warmup_epochs: 3,
    mosaic: 0.8,
    mixup: 0.1,
    close_mosaic: 20,
    amp: true
  },
  output: {
    project: "runs/distill",
    name: "adaptive_kd_v1"
  }
};

function useToast() {
  const [toasts, setToasts] = useState([]);
  const push = (message, type = "info") => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setToasts((prev) => [...prev, { id, message, type }]);
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3000);
  };
  return { toasts, push };
}

async function apiRequest(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options
  });
  const data = await response.json();
  if (!response.ok) {
    const err = new Error(data.error || "请求失败");
    err.status = response.status;
    err.payload = data;
    throw err;
  }
  return data;
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const h = Math.floor(safe / 3600);
  const m = Math.floor((safe % 3600) / 60);
  const s = safe % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function detectLogLevel(line) {
  const text = String(line || "");
  if (/\b(error|exception|traceback|failed?)\b/i.test(text)) return "error";
  if (/(\bwarn(ing)?\b|caution|警告|告警|⚠|\[W\]|^\s*W\d*:|\bignoring\b|忽略|已忽略|\bdeprecated\b)/i.test(text)) return "warning";
  if (/\b(success|done|completed?)\b/i.test(text)) return "success";
  return "info";
}

function App() {
  const [activeTab, setActiveTab] = useState("training");
  const [theme, setTheme] = useState(() => window.localStorage.getItem("edgedistilldet-theme") || "light");
  const { toasts, push } = useToast();

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("edgedistilldet-theme", theme);
  }, [theme]);

  return (
    <>
      <header className="md-app-bar">
        <div className="app-bar-content">
          <span className="material-icons app-icon">radar</span>
          <h1 className="app-title">EdgeDistillDet</h1>
          <span className="app-subtitle">边缘蒸馏训练工作台 (React 迁移中)</span>
          <div className="app-bar-spacer" />
          <button
            className="theme-toggle"
            title="切换明暗主题"
            onClick={() => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
          >
            <span className="material-icons icon-sun">light_mode</span>
            <span className="material-icons icon-moon">dark_mode</span>
          </button>
          <div className="status-indicator online">
            <span className="status-dot" />
            <span>服务正常</span>
          </div>
        </div>
      </header>

      <main className="md-main">
        <nav className="tab-nav" role="tablist">
          <button className={`tab-btn ${activeTab === "training" ? "active" : ""}`} onClick={() => setActiveTab("training")}>
            <span className="material-icons tab-icon">tune</span>
            训练配置
          </button>
          <button className={`tab-btn ${activeTab === "metrics" ? "active" : ""}`} onClick={() => setActiveTab("metrics")}>
            <span className="material-icons tab-icon">analytics</span>
            指标监控
          </button>
          <button className={`tab-btn ${activeTab === "agent" ? "active" : ""}`} onClick={() => setActiveTab("agent")}>
            <span className="material-icons tab-icon">smart_toy</span>
            Agent
          </button>
        </nav>

        <div className="tab-panels">
          {activeTab === "training" ? <TrainingPanel toast={push} /> : null}
          {activeTab === "metrics" ? <MetricsPanel toast={push} /> : null}
          {activeTab === "agent" ? <AgentPanel toast={push} /> : null}
        </div>
      </main>

      <div id="toast-container" className="toast-container">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            <span className="material-icons">
              {toast.type === "success" ? "check_circle" : toast.type === "error" ? "error" : "info"}
            </span>
            <span>{toast.message}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function TrainingPanel({ toast }) {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [mode, setMode] = useState("distill");
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [resumeCandidates, setResumeCandidates] = useState([]);
  const [selectedResumeIndex, setSelectedResumeIndex] = useState(0);
  const [runHint, setRunHint] = useState("将根据项目目录自动推荐可用运行名称。");
  const [outputCheckInfo, setOutputCheckInfo] = useState({ project: "runs/distill", existingNames: [], suggested: "exp1" });
  const [progress, setProgress] = useState({ current: 0, total: 0, elapsed: "--:--:--", expected: "--:--:--" });
  const [autoScroll, setAutoScroll] = useState(true);
  const [quickStats, setQuickStats] = useState({
    loss: "--",
    kd: "--",
    alpha: "--",
    temp: "--",
    map50: "--",
    map95: "--",
    lr: "--"
  });

  const logOffsetRef = useRef(0);
  const startTimestampRef = useRef(null);
  const logContainerRef = useRef(null);
  const overlapAlertShownRef = useRef("");

  const setNested = (scope, key, value) => {
    setForm((prev) => ({
      ...prev,
      [scope]: { ...prev[scope], [key]: value }
    }));
  };

  const pickLocalPath = async ({ kind = "file", title = "选择路径", initialPath = "", filters = [] } = {}) => {
    const result = await apiRequest("/api/dialog/pick", {
      method: "POST",
      body: JSON.stringify({
        kind,
        title,
        initial_path: initialPath || "",
        filters
      })
    });
    return String(result?.path || "");
  };

  const mergeConfig = (config) => {
    setForm((prev) => ({
      ...prev,
      ...config,
      distillation: { ...prev.distillation, ...config?.distillation },
      training: { ...prev.training, ...config?.training },
      output: { ...prev.output, ...config?.output }
    }));
  };

  const fetchDefaultConfig = async () => {
    try {
      const data = await apiRequest("/api/config/distill_config.yaml");
      mergeConfig(data.config || {});
    } catch {
      // 使用默认配置
    }
  };

  const refreshRunNameSuggestion = async (project, currentName, forceDefault = false) => {
    try {
      const params = new URLSearchParams({ project });
      const info = await apiRequest(`/api/output/check?${params.toString()}`);
      const suggested = info.next_exp_name || "exp1";
      const existingNames = Array.isArray(info.existing_names) ? info.existing_names : [];
      const shouldReplace = forceDefault || !currentName || currentName === "adaptive_kd_v1" || currentName === "adaptive_kd";
      setOutputCheckInfo({
        project: info.project || project,
        existingNames,
        suggested
      });
      setForm((prev) => ({
        ...prev,
        output: {
          ...prev.output,
          name: shouldReplace ? suggested : prev.output.name
        }
      }));
      setRunHint(`建议输出目录: ${project}/${suggested}`);
    } catch {
      setRunHint("无法获取运行名称建议。");
    }
  };

  const refreshResumeCandidates = async (project, autoSelect = true) => {
    try {
      const params = new URLSearchParams({ project });
      const result = await apiRequest(`/api/train/resume_candidates?${params.toString()}`);
      const candidates = Array.isArray(result.candidates) ? result.candidates : [];
      setResumeCandidates(candidates);
      if (autoSelect && candidates.length) {
        setSelectedResumeIndex(0);
        const c = candidates[0];
        setForm((prev) => ({
          ...prev,
          output: { ...prev.output, project: c.project, name: c.name }
        }));
      }
    } catch {
      setResumeCandidates([]);
    }
  };

  useEffect(() => {
    fetchDefaultConfig().then(() => {
      refreshRunNameSuggestion(DEFAULT_FORM.output.project, DEFAULT_FORM.output.name, true);
      refreshResumeCandidates(DEFAULT_FORM.output.project);
    });
  }, []);

  useEffect(() => {
    const statusTimer = window.setInterval(async () => {
      try {
        const data = await apiRequest("/api/train/status");
        const nextRunning = Boolean(data.running);
        setRunning(nextRunning);
        if (nextRunning) {
          if (!startTimestampRef.current && data.start_time) {
            startTimestampRef.current = Math.floor(Number(data.start_time) * 1000);
          }
          const currentEpoch = Number(data.current_epoch) || 0;
          const totalEpoch = Number(data.total_epochs) || 0;
          const now = Date.now();
          const started = startTimestampRef.current || now;
          const elapsedSec = Math.floor((now - started) / 1000);
          const expectedSec = currentEpoch > 0 && totalEpoch > 0 ? Math.round((elapsedSec / currentEpoch) * totalEpoch) : 0;
          setProgress({
            current: currentEpoch,
            total: totalEpoch,
            elapsed: formatTime(elapsedSec),
            expected: expectedSec > 0 ? formatTime(expectedSec) : "--:--:--"
          });
        } else if (!nextRunning) {
          startTimestampRef.current = null;
        }
      } catch {
        // 静默处理
      }
    }, 2000);
    return () => window.clearInterval(statusTimer);
  }, []);

  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setInterval(async () => {
      try {
        const offset = Number.isFinite(logOffsetRef.current) ? logOffsetRef.current : 0;
        const data = await apiRequest(`/api/train/logs?offset=${offset}&limit=120`);
        if (!Array.isArray(data.logs) || data.logs.length === 0) return;
        logOffsetRef.current = data.offset + data.logs.length;
        setLogs((prev) => [...prev, ...data.logs].slice(-800));
        parseMetricsFromLogLines(data.logs, setQuickStats, setProgress);
      } catch {
        // 静默处理
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [running]);

  useEffect(() => {
    const el = logContainerRef.current;
    if (!el) return;
    if (!autoScroll) return;
    window.requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
  }, [logs, autoScroll]);

  const currentOutputProject = outputCheckInfo.project || form.output.project || "runs/distill";
  const currentOutputName = (form.output.name || "").trim();
  const isOutputPathOverlap = Boolean(currentOutputName && outputCheckInfo.existingNames.includes(currentOutputName));
  const renderedHint = isOutputPathOverlap
    ? `路径重合: ${currentOutputProject}/${currentOutputName}`
    : `建议输出目录: ${currentOutputProject}/${outputCheckInfo.suggested || "exp1"}`;

  useEffect(() => {
    if (!isOutputPathOverlap || !currentOutputName) return;
    const overlapKey = `${currentOutputProject}/${currentOutputName}`;
    if (overlapAlertShownRef.current === overlapKey) return;
    overlapAlertShownRef.current = overlapKey;
    window.alert(`路径重合：${overlapKey}`);
    toast(`路径重合：${overlapKey}`, "warning");
  }, [currentOutputProject, currentOutputName, isOutputPathOverlap]);

  const saveConfig = async () => {
    await apiRequest("/api/config/save", {
      method: "POST",
      body: JSON.stringify({ name: "distill_config.yaml", config: form })
    });
  };

  const startTraining = async () => {
    const studentWeight = form.distillation.student_weight?.trim();
    const teacherWeight = form.distillation.teacher_weight?.trim();
    const dataYaml = form.training.data_yaml?.trim();
    if (!studentWeight) return toast("请选择学生模型权重文件", "warning");
    if (!dataYaml) return toast("请填写数据集配置文件路径", "warning");
    if (mode !== "resume" && !teacherWeight) return toast("请选择教师模型权重文件", "warning");
    if (mode === "resume" && resumeCandidates.length === 0) {
      return toast("当前没有可用断点，请先完成一次训练或切换到蒸馏训练", "warning");
    }

    try {
      await saveConfig();
      const body = { config: "distill_config.yaml", mode };
      if (mode === "resume" && resumeCandidates[selectedResumeIndex]?.checkpoint) {
        body.checkpoint = resumeCandidates[selectedResumeIndex].checkpoint;
      }
      await apiRequest("/api/train/start", { method: "POST", body: JSON.stringify(body) });
      startTimestampRef.current = Date.now();
      logOffsetRef.current = 0;
      setLogs([]);
      setRunning(true);
      toast("训练任务已启动", "success");
    } catch (error) {
      const requiresConfirmation = Boolean(error?.status === 409 && error?.payload?.requires_confirmation);
      if (requiresConfirmation) {
        const project = error.payload?.project || form.output?.project || "runs/distill";
        const name = error.payload?.name || form.output?.name || "exp";
        const confirmed = window.confirm(`输出目录 ${project}/${name} 已存在，是否继续覆盖？`);
        if (!confirmed) return;
        const overwriteBody = { config: "distill_config.yaml", mode, allow_overwrite: true };
        if (mode === "resume" && resumeCandidates[selectedResumeIndex]?.checkpoint) {
          overwriteBody.checkpoint = resumeCandidates[selectedResumeIndex].checkpoint;
        }
        await apiRequest("/api/train/start", { method: "POST", body: JSON.stringify(overwriteBody) });
        startTimestampRef.current = Date.now();
        logOffsetRef.current = 0;
        setLogs([]);
        setRunning(true);
        toast("训练任务已启动（已确认覆盖）", "success");
        return;
      }
      toast(error.message, "error");
    }
  };

  const stopTraining = async () => {
    if (!running) {
      toast("当前没有运行中的训练任务", "info");
      return;
    }
    if (!window.confirm("确定要停止当前训练吗？")) return;
    try {
      await apiRequest("/api/train/stop", { method: "POST", body: JSON.stringify({}) });
      setRunning(false);
      toast("训练已停止", "info");
    } catch (error) {
      toast(error.message, "error");
    }
  };

  const loadConfigFromFile = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".yaml") && !file.name.toLowerCase().endsWith(".yml")) {
      toast("请选择 YAML 文件", "warning");
      return;
    }
    try {
      const content = await file.text();
      const result = await apiRequest("/api/config/upload", {
        method: "POST",
        body: JSON.stringify({ content, name: file.name })
      });
      mergeConfig(result.config || {});
      toast(`已加载本地配置: ${file.name}`, "success");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      event.target.value = "";
    }
  };

  const resetForm = async () => {
    if (!window.confirm("确定要重置所有参数到最近保存的配置吗?")) return;
    try {
      const result = await apiRequest("/api/config/recent");
      mergeConfig(result.config || {});
      toast("已从最近保存的配置重置表单", "success");
    } catch (error) {
      toast(error.message, "error");
    }
  };

  const clearLogs = () => {
    setLogs([]);
    logOffsetRef.current = Number.MAX_SAFE_INTEGER;
  };

  const downloadLogs = async () => {
    try {
      const response = await fetch("/api/train/logs/download");
      if (!response.ok) throw new Error("下载日志失败");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `training_logs_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.txt`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      toast("训练日志已下载", "success");
    } catch (error) {
      toast(error.message, "error");
    }
  };

  const progressPercent = progress.total > 0 ? Math.min(100, (progress.current / progress.total) * 100) : 0;
  const isResumeStartDisabled = mode === "resume" && resumeCandidates.length === 0;

  return (
    <div className="tab-panel active" id="panel-training">
      <section className="train-launcher">
        <div className="launcher-left">
          <div className="launch-info">
            <div className="launch-header">
              <h2><span className="material-icons">rocket_launch</span> 训练控制台</h2>
              <span className={`badge ${running ? "running" : "idle"}`}>{running ? "训练中" : "就绪"}</span>
            </div>
            <p className="launch-desc">配置参数后选择训练模式并启动训练任务</p>
          </div>

          <div className="launch-modes">
            <label className={`mode-option ${mode === "distill" ? "selected" : ""}`}>
              <input type="radio" checked={mode === "distill"} onChange={() => setMode("distill")} disabled={running} />
              <div className="mode-card">
                <span className="material-icons mode-icon">school</span>
                <div className="mode-text">
                  <strong>蒸馏训练</strong>
                  <span>知识蒸馏训练，训练完成后自动评估模型性能</span>
                </div>
              </div>
            </label>
            <label className={`mode-option ${mode === "resume" ? "selected" : ""}`}>
              <input type="radio" checked={mode === "resume"} onChange={() => setMode("resume")} disabled={running} />
              <div className="mode-card">
                <span className="material-icons mode-icon">restart_alt</span>
                <div className="mode-text">
                  <strong>断点续训</strong>
                  <span>从上次检查点恢复训练进度</span>
                </div>
              </div>
            </label>
          </div>

          <div className="launch-actions">
            <button
              id="btn-start-training"
              className="btn-start"
              onClick={startTraining}
              disabled={running || isResumeStartDisabled}
              title={isResumeStartDisabled ? "当前没有可用断点，无法开始断点续训" : ""}
            >
              <span className="material-icons">play_arrow</span>开始训练
            </button>
            <button id="btn-stop-training" className="btn-stop" onClick={stopTraining} disabled={!running}>
              <span className="material-icons">stop</span>停止训练
            </button>
            <label className="md-btn md-btn-tonal" style={{ cursor: "pointer" }}>
              <span className="material-icons">file_open</span>加载配置
              <input type="file" accept=".yaml,.yml" style={{ display: "none" }} onChange={loadConfigFromFile} />
            </label>
            <button className="md-btn md-btn-outlined" onClick={() => saveConfig().then(() => toast("配置已保存", "success")).catch((e) => toast(e.message, "error"))}>
              <span className="material-icons">save</span>保存配置
            </button>
            <button className="md-btn md-btn-text" onClick={resetForm}>
              <span className="material-icons">refresh</span>重置表单
            </button>
          </div>
        </div>

        <div className="launcher-side">
          <div className="config-card launcher-side-card">
            <h3 className="card-header">输出配置</h3>
            <div className="form-row stacked-row">
              <div className="form-group">
                <label>项目目录</label>
                <PathField
                  value={form.output.project || ""}
                  onChange={(project) => {
                    setNested("output", "project", project);
                    refreshRunNameSuggestion(project || "runs/distill", form.output.name, true);
                    refreshResumeCandidates(project || "runs/distill", false);
                  }}
                  onBrowse={async () => {
                    try {
                      const selected = await pickLocalPath({
                        kind: "directory",
                        title: "选择训练输出项目目录",
                        initialPath: form.output.project || "runs/distill"
                      });
                      if (!selected) return;
                      setNested("output", "project", selected);
                      refreshRunNameSuggestion(selected || "runs/distill", form.output.name, true);
                      refreshResumeCandidates(selected || "runs/distill", false);
                    } catch (error) {
                      toast(error.message, "error");
                    }
                  }}
                  disabled={running}
                />
              </div>
              <div className="form-group">
                <label>运行名称</label>
                <input
                  className="md-input"
                  value={form.output.name || ""}
                  onChange={(e) => {
                    const nextName = e.target.value;
                    setNested("output", "name", nextName);
                  }}
                  disabled={running}
                />
                <small className={`hint ${isOutputPathOverlap ? "warning" : ""}`}>{renderedHint || runHint}</small>
              </div>
            </div>
          </div>
          <div className={`config-card launcher-side-card ${mode !== "resume" ? "disabled-panel" : ""}`}>
            <h3 className="card-header">续训历史</h3>
            <div className="form-group">
              <label>请选择历史运行</label>
              <select
                className="md-select"
                value={String(selectedResumeIndex)}
                onChange={(e) => {
                  const idx = Number(e.target.value) || 0;
                  setSelectedResumeIndex(idx);
                  const c = resumeCandidates[idx];
                  if (c) {
                    setForm((prev) => ({
                      ...prev,
                      output: { ...prev.output, project: c.project, name: c.name }
                    }));
                  }
                }}
                disabled={mode !== "resume" || running || resumeCandidates.length === 0}
              >
                {resumeCandidates.length === 0 ? <option value="0">暂无可用候选</option> : null}
                {resumeCandidates.map((item, idx) => (
                  <option key={`${item.project}-${item.name}-${idx}`} value={idx}>{item.display_name}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </section>

      <div className="panel-grid">
        <div className="config-section">
          <div className="config-card">
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
                  disabled={running}
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
                  disabled={running}
                />
              </div>
            </div>
          </div>

          <div className="config-card">
            <h3 className="card-header">蒸馏核心参数</h3>
            <div className="form-grid">
              <NumberField label="初始 Alpha 权重" value={form.distillation.alpha_init} step="0.01" onChange={(v) => setNested("distillation", "alpha_init", v)} disabled={running} />
              <NumberField label="温度上限 T_max" value={form.distillation.T_max} step="0.1" onChange={(v) => setNested("distillation", "T_max", v)} disabled={running} />
              <NumberField label="温度下限 T_min" value={form.distillation.T_min} step="0.1" onChange={(v) => setNested("distillation", "T_min", v)} disabled={running} />
              <NumberField label="预热轮数" value={form.distillation.warm_epochs} step="1" onChange={(v) => setNested("distillation", "warm_epochs", v)} disabled={running} />
              <NumberField label="KD 损失权重 w_kd" value={form.distillation.w_kd} step="0.01" onChange={(v) => setNested("distillation", "w_kd", v)} disabled={running} />
              <NumberField label="Focal KD 权重 w_focal" value={form.distillation.w_focal} step="0.01" onChange={(v) => setNested("distillation", "w_focal", v)} disabled={running} />
              <NumberField label="特征对齐权重 w_feat" value={form.distillation.w_feat} step="0.01" onChange={(v) => setNested("distillation", "w_feat", v)} disabled={running} />
              <NumberField label="小目标增强系数" value={form.distillation.scale_boost} step="0.1" onChange={(v) => setNested("distillation", "scale_boost", v)} disabled={running} />
              <NumberField label="Focal Gamma 参数" value={form.distillation.focal_gamma} step="0.1" onChange={(v) => setNested("distillation", "focal_gamma", v)} disabled={running} />
            </div>
          </div>

          <div className="config-card">
            <h3 className="card-header">训练超参数</h3>
            <div className="form-grid">
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
                disabled={running}
              />
              <SelectField
                label="设备"
                value={form.training.device}
                onChange={(v) => setNested("training", "device", v)}
                options={[
                  { value: "0", label: "GPU 0" },
                  { value: "1", label: "GPU 1" },
                  { value: "cpu", label: "CPU" }
                ]}
                disabled={running}
              />
              <NumberField label="训练轮数 Epochs" value={form.training.epochs} step="1" onChange={(v) => setNested("training", "epochs", v)} disabled={running} />
              <NumberField label="图像尺寸 imgsz" value={form.training.imgsz} step="1" onChange={(v) => setNested("training", "imgsz", v)} disabled={running} />
              <NumberField label="Batch Size" value={form.training.batch} step="1" onChange={(v) => setNested("training", "batch", v)} disabled={running} />
              <NumberField label="数据加载线程数" value={form.training.workers} step="1" onChange={(v) => setNested("training", "workers", v)} disabled={running} />
              <NumberField label="初始学习率 lr0" value={form.training.lr0} step="0.001" onChange={(v) => setNested("training", "lr0", v)} disabled={running} />
              <NumberField label="最终学习率因子 lrf" value={form.training.lrf} step="0.01" onChange={(v) => setNested("training", "lrf", v)} disabled={running} />
              <NumberField label="学习率预热轮数" value={form.training.warmup_epochs} step="0.5" onChange={(v) => setNested("training", "warmup_epochs", v)} disabled={running} />
              <NumberField label="Mosaic 增强概率" value={form.training.mosaic} step="0.01" onChange={(v) => setNested("training", "mosaic", v)} disabled={running} />
              <NumberField label="Mixup 增强概率" value={form.training.mixup} step="0.01" onChange={(v) => setNested("training", "mixup", v)} disabled={running} />
              <NumberField label="关闭 Mosaic 的 epoch" value={form.training.close_mosaic} step="1" onChange={(v) => setNested("training", "close_mosaic", v)} disabled={running} />
              <div className="form-group switch-group">
                <label>AMP 混合精度训练</label>
                <label className="md-switch">
                  <input type="checkbox" checked={Boolean(form.training.amp)} onChange={(e) => setNested("training", "amp", e.target.checked)} disabled={running} />
                </label>
              </div>
            </div>
          </div>
        </div>

        <div className="log-section">
          <div className="log-card">
            <div className="log-header">
              <h3><span className="material-icons">terminal</span>训练日志</h3>
              <div className="log-controls">
                <span className={`badge ${running ? "running" : "idle"}`}>{running ? "训练中" : "空闲"}</span>
                <button className="btn-icon-sm" onClick={clearLogs} title="清空日志"><span className="material-icons">delete_outline</span></button>
                <button className="btn-icon-sm" onClick={downloadLogs} title="下载日志"><span className="material-icons">download</span></button>
                <button
                  className={`btn-icon-sm ${!autoScroll ? "is-disabled" : ""}`}
                  onClick={() => setAutoScroll((prev) => !prev)}
                  title="自动滚动"
                >
                  <span className="material-icons">vertical_align_bottom</span>
                </button>
              </div>
            </div>

            <div className="progress-container" style={{ display: running ? "block" : "none" }}>
              <div className="progress-bar-wrapper">
                <div className="progress-bar" style={{ width: `${progressPercent.toFixed(1)}%` }} />
              </div>
              <div className="progress-info">
                <span>{`Epoch: ${progress.current} / ${progress.total || "-"}`}</span>
                <span>{`耗时: ${progress.elapsed}`}</span>
                <span>{`预计总耗时: ${progress.expected}`}</span>
              </div>
            </div>

            <div ref={logContainerRef} className="log-container">
              {logs.length === 0 ? <div className="log-line info">暂无日志输出</div> : null}
              {logs.map((line, index) => {
                const level = detectLogLevel(line);
                return (
                  <div key={`${index}-${String(line).slice(0, 18)}`} className={`log-line ${level}`}>
                    {String(line)}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="quick-stats" style={{ display: quickStats.loss === "--" ? "none" : "grid" }}>
            <QuickStat label="当前 Loss" value={quickStats.loss} />
            <QuickStat label="KD Loss" value={quickStats.kd} />
            <QuickStat label="Alpha 权重" value={quickStats.alpha} />
            <QuickStat label="温度 T" value={quickStats.temp} />
            <QuickStat label="mAP50" value={quickStats.map50} />
            <QuickStat label="mAP50-95" value={quickStats.map95} />
            <QuickStat label="学习率" value={quickStats.lr} />
          </div>
        </div>
      </div>
    </div>
  );
}

function parseMetricsFromLogLines(lines, setQuickStats, setProgress) {
  const last = Array.isArray(lines) ? lines.slice(-20) : [];
  last.forEach((line) => {
    const text = String(line);
    const ep = text.match(/\[EPOCH_PROGRESS\]\s+epoch=(\d+)\s+total=(\d+)\s+loss=([\d.]+)\s+kd=([\d.]+)\s+alpha=([\d.]+)\s+temp=([\d.]+)/i);
    if (ep) {
      setProgress((prev) => ({ ...prev, current: Number(ep[1]), total: Number(ep[2]) }));
      setQuickStats((prev) => ({
        ...prev,
        loss: ep[3],
        kd: ep[4],
        alpha: ep[5],
        temp: ep[6]
      }));
    }
    const map = text.match(/mAP50=([\d.]+)\s+mAP50-95=([\d.]+)/i);
    if (map) {
      setQuickStats((prev) => ({ ...prev, map50: map[1], map95: map[2] }));
    }
    const lr = text.match(/lr0=([\d.eE-]+)/i);
    if (lr) {
      setQuickStats((prev) => ({ ...prev, lr: lr[1] }));
    }
  });
}

function QuickStat({ label, value }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

function TextField({ label, value, onChange, disabled }) {
  return (
    <div className="form-group">
      <label>{label}</label>
      <input className="md-input" value={value || ""} onChange={(e) => onChange(e.target.value)} disabled={disabled} />
    </div>
  );
}

function PathField({ label, value, onChange, onBrowse, disabled }) {
  return (
    <div className="form-group">
      {label ? <label>{label}</label> : null}
      <div className="file-input-wrapper">
        <input className="md-input" value={value || ""} onChange={(e) => onChange(e.target.value)} disabled={disabled} />
        <button type="button" className="file-picker-label" onClick={onBrowse} disabled={disabled} title="浏览本地路径">
          <span className="material-icons">folder_open</span>
        </button>
      </div>
    </div>
  );
}

function NumberField({ label, value, onChange, step, disabled }) {
  return (
    <div className="form-group">
      <label>{label}</label>
      <input className="md-input" type="number" step={step} value={value ?? ""} onChange={(e) => onChange(Number(e.target.value))} disabled={disabled} />
    </div>
  );
}

function SelectField({ label, value, onChange, options, disabled }) {
  return (
    <div className="form-group">
      <label>{label}</label>
      <select className="md-select" value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
}

function MetricsPanel({ toast }) {
  const [sources, setSources] = useState([]);
  const [source, setSource] = useState("");
  const [overview, setOverview] = useState({});
  const [summaryMetrics, setSummaryMetrics] = useState({});
  const [hasData, setHasData] = useState(false);
  const [chartType, setChartType] = useState("all");
  const [lossRange, setLossRange] = useState("all");
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshLeft, setRefreshLeft] = useState(30);
  const [chartSeriesState, setChartSeriesState] = useState(null);
  const [themeMode, setThemeMode] = useState(() => document.documentElement.getAttribute("data-theme") || "light");

  const lossRef = useRef(null);
  const mapRef = useRef(null);
  const lrRef = useRef(null);
  const distillRef = useRef(null);
  const prRef = useRef(null);
  const classRef = useRef(null);
  const chartInstances = useRef({});
  const rawSeriesRef = useRef(null);

  const refreshSources = async (showToast = false) => {
    try {
      const data = await apiRequest("/api/metrics");
      const available = Array.isArray(data.csv_metrics) ? data.csv_metrics.filter((x) => x.has_results) : [];
      setSources(available);
      if (!available.length) {
        setHasData(false);
        if (showToast) toast("暂无训练结果可展示", "info");
        return;
      }
      const nextSource = available.some((it) => it.path === source) ? source : available[0].path;
      setSource(nextSource);
      if (showToast) toast("指标数据已刷新", "success");
    } catch (error) {
      toast(error.message, "error");
    }
  };

  const loadMetricsData = async (sourcePath, silent = false) => {
    if (!sourcePath) return;
    try {
      const data = await apiRequest(`/api/metrics?source=${encodeURIComponent(sourcePath)}`);
      const epochs = data.chart_series?.epochs || [];
      setHasData(epochs.length > 0);
      setOverview(data.overview_stats || {});
      setSummaryMetrics(data.summary_metrics || {});
      const nextSeries = data.chart_series || null;
      rawSeriesRef.current = nextSeries;
      setChartSeriesState(nextSeries);
      if (!silent) toast("图表已更新", "success");
    } catch (error) {
      setHasData(false);
      setChartSeriesState(null);
      toast(error.message, "error");
    }
  };

  useEffect(() => {
    refreshSources();
  }, []);

  useEffect(() => {
    if (!source) return;
    loadMetricsData(source, true);
  }, [source]);

  useEffect(() => {
    if (!rawSeriesRef.current) return;
    renderAllCharts(chartInstances, {
      lossRef,
      mapRef,
      lrRef,
      distillRef,
      prRef,
      classRef
    }, rawSeriesRef.current, lossRange);
  }, [lossRange]);

  useEffect(() => {
    if (!hasData || !chartSeriesState) return;
    renderAllCharts(chartInstances, {
      lossRef,
      mapRef,
      lrRef,
      distillRef,
      prRef,
      classRef
    }, chartSeriesState, lossRange);
  }, [hasData, chartSeriesState, lossRange, themeMode]);

  useEffect(() => {
    const root = document.documentElement;
    const observer = new MutationObserver(() => {
      const nextTheme = root.getAttribute("data-theme") || "light";
      setThemeMode((prev) => (prev === nextTheme ? prev : nextTheme));
    });
    observer.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!autoRefresh || !source) return undefined;
    setRefreshLeft(30);
    const tick = window.setInterval(() => {
      setRefreshLeft((prev) => {
        if (prev <= 1) return 30;
        return prev - 1;
      });
    }, 1000);
    const reload = window.setInterval(() => {
      loadMetricsData(source, true);
    }, 30000);
    return () => {
      window.clearInterval(tick);
      window.clearInterval(reload);
    };
  }, [autoRefresh, source]);

  useEffect(() => () => {
    Object.values(chartInstances.current).forEach((chart) => chart.destroy());
  }, []);

  const exportChart = (canvasRef, fileName) => {
    const canvas = canvasRef?.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = `${fileName}_${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  const exportTable = () => {
    const entries = Object.entries(summaryMetrics || {});
    if (!entries.length) {
      toast("暂无可导出数据", "warning");
      return;
    }
    let csv = "\uFEFF指标,最佳值,最终值,改善幅度,趋势\n";
    entries.forEach(([name, val]) => {
      csv += `${name},${val.best},${val.final},${val.improvement || ""},${val.trend || ""}\n`;
    });
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `training_results_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const showLoss = chartType === "all" || chartType === "loss";
  const showAcc = chartType === "all" || chartType === "accuracy";
  const showLr = chartType === "all" || chartType === "lr";

  return (
    <div className="tab-panel active" id="panel-metrics">
      <div className="metrics-toolbar">
        <div className="toolbar-left">
          <button className="md-btn md-btn-outlined" onClick={() => refreshSources(true)}>
            <span className="material-icons">refresh</span>刷新数据
          </button>
          <select className="md-select" value={source} onChange={(e) => setSource(e.target.value)}>
            {sources.length === 0 ? <option value="">暂无训练结果可选</option> : null}
            {sources.map((item) => (
              <option key={item.path} value={item.path}>
                {item.display_name || item.name}
              </option>
            ))}
          </select>
        </div>
        <div className="auto-refresh-bar">
          <label className="auto-refresh-label">
            <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
            自动刷新图表数据
          </label>
          <span className="auto-refresh-interval">{autoRefresh ? `${refreshLeft}s 后刷新` : "--"}</span>
        </div>
        <div className="toolbar-right">
          <div className="chip-group">
            <button className={`chip ${chartType === "loss" ? "active" : ""}`} onClick={() => setChartType("loss")}>损失</button>
            <button className={`chip ${chartType === "accuracy" ? "active" : ""}`} onClick={() => setChartType("accuracy")}>精度</button>
            <button className={`chip ${chartType === "lr" ? "active" : ""}`} onClick={() => setChartType("lr")}>学习率</button>
            <button className={`chip ${chartType === "all" ? "active" : ""}`} onClick={() => setChartType("all")}>全部</button>
          </div>
        </div>
      </div>

      {!hasData ? (
        <div className="metrics-empty">
          <div className="empty-placeholder">
            <span className="material-icons">analytics</span>
            <p>暂无训练结果可展示。请先执行训练，并刷新后查看指标监控。</p>
          </div>
        </div>
      ) : (
        <div id="metrics-content">
          <div className="metrics-overview">
            <OverviewCard label="最佳 mAP@50" value={overview["ov-map50"] || "--"} icon="trending_up" />
            <OverviewCard label="推理 FPS (GPU)" value={overview["ov-fps"] || "--"} icon="speed" />
            <OverviewCard label="模型参数量" value={overview["ov-params"] || "--"} icon="memory" />
            <OverviewCard label="训练总耗时" value={overview["ov-time"] || "--"} icon="timer" />
          </div>

          <div className="charts-grid">
            <div className="chart-card wide" style={{ display: showLoss ? "" : "none" }}>
              <div className="chart-header">
                <h3>训练损失曲线</h3>
                <div className="chart-actions">
                  <select className="md-select mini-select" value={lossRange} onChange={(e) => setLossRange(e.target.value)}>
                    <option value="all">全部 Epochs</option>
                    <option value="last30">最近 30</option>
                    <option value="last10">最近 10</option>
                  </select>
                  <button className="btn-icon-sm" onClick={() => exportChart(lossRef, "loss-chart")}><span className="material-icons">download</span></button>
                </div>
              </div>
              <div className="chart-body"><canvas ref={lossRef} /></div>
            </div>

            <div className="chart-card" style={{ display: showAcc ? "" : "none" }}>
              <div className="chart-header">
                <h3>mAP 曲线</h3>
                <button className="btn-icon-sm" onClick={() => exportChart(mapRef, "map-chart")}><span className="material-icons">download</span></button>
              </div>
              <div className="chart-body"><canvas ref={mapRef} /></div>
            </div>

            <div className="chart-card" style={{ display: showLr ? "" : "none" }}>
              <div className="chart-header">
                <h3>学习率变化</h3>
                <button className="btn-icon-sm" onClick={() => exportChart(lrRef, "lr-chart")}><span className="material-icons">download</span></button>
              </div>
              <div className="chart-body"><canvas ref={lrRef} /></div>
            </div>

            <div className="chart-card wide" style={{ display: showAcc ? "" : "none" }}>
              <div className="chart-header"><h3>蒸馏指标 (Alpha/Temperature 动态变化)</h3></div>
              <div className="chart-body"><canvas ref={distillRef} /></div>
            </div>

            <div className="chart-card" style={{ display: showAcc ? "" : "none" }}>
              <div className="chart-header"><h3>Precision-Recall 曲线</h3></div>
              <div className="chart-body"><canvas ref={prRef} /></div>
            </div>

            <div className="chart-card" style={{ display: showAcc ? "" : "none" }}>
              <div className="chart-header"><h3>各类别性能分布</h3></div>
              <div className="chart-body"><canvas ref={classRef} /></div>
            </div>

            <div className="chart-card full-width">
              <div className="chart-header">
                <h3>训练结果摘要</h3>
                <button className="md-btn md-btn-tonal sm-btn" onClick={exportTable}>
                  <span className="material-icons">table_chart</span>导出表格
                </button>
              </div>
              <div className="table-container">
                <table className="md-table">
                  <thead>
                    <tr><th>指标</th><th>最佳值</th><th>最终值</th><th>改善幅度</th><th>趋势</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(summaryMetrics).length === 0 ? (
                      <tr><td colSpan={5} className="empty-hint">暂无数据，请先完成一次训练</td></tr>
                    ) : (
                      Object.entries(summaryMetrics).map(([metric, val]) => (
                        <tr key={metric}>
                          <td>{metric}</td>
                          <td>{Number(val.best).toFixed(4)}</td>
                          <td>{Number(val.final).toFixed(4)}</td>
                          <td>{val.improvement || "--"}</td>
                          <td>{val.trend || "stable"}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function OverviewCard({ icon, label, value }) {
  return (
    <div className="overview-card">
      <div className="overview-icon"><span className="material-icons">{icon}</span></div>
      <div className="overview-info">
        <span className="overview-label">{label}</span>
        <span className="overview-value">{value}</span>
      </div>
    </div>
  );
}

function getChartTheme() {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  return {
    text: isDark ? "rgba(230, 237, 243, 0.92)" : "rgba(28, 27, 31, 0.85)",
    muted: isDark ? "rgba(150, 160, 175, 0.78)" : "rgba(98, 91, 113, 0.75)",
    grid: isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(103, 80, 164, 0.10)",
    tooltipBg: isDark ? "rgba(17, 22, 28, 0.96)" : "rgba(255, 255, 255, 0.96)",
    tooltipBorder: isDark ? "rgba(80, 90, 108, 0.5)" : "rgba(103, 80, 164, 0.25)"
  };
}

function renderLineChart(instancesRef, canvas, key, labels, datasets, options = {}) {
  if (!canvas) return;
  if (instancesRef.current[key]) instancesRef.current[key].destroy();
  try {
    const theme = getChartTheme();
    instancesRef.current[key] = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: datasets.map((item) => ({
          label: item.label,
          data: item.data,
          borderColor: item.color,
          backgroundColor: `${item.color}20`,
          fill: false,
          borderWidth: 2.5,
          tension: 0.38,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHitRadius: 10
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false
        },
        animation: {
          duration: 520,
          easing: "easeOutQuart"
        },
        elements: {
          line: {
            capBezierPoints: true
          }
        },
        scales: {
          x: {
            grid: { color: theme.grid, drawBorder: false },
            ticks: { color: theme.muted, maxTicksLimit: 10 }
          },
          y: {
            grid: { color: theme.grid, drawBorder: false },
            ticks: { color: theme.muted }
          }
        },
        plugins: {
          legend: {
            position: "top",
            labels: {
              color: theme.text,
              usePointStyle: true,
              pointStyle: "circle",
              boxWidth: 8,
              boxHeight: 8,
              padding: 14
            }
          },
          tooltip: {
            backgroundColor: theme.tooltipBg,
            borderColor: theme.tooltipBorder,
            borderWidth: 1,
            titleColor: theme.text,
            bodyColor: theme.text,
            displayColors: true,
            padding: 10
          }
        },
        ...options
      }
    });
  } catch {}
}

function renderBarChart(instancesRef, canvas, key, labels, datasets) {
  if (!canvas) return;
  if (instancesRef.current[key]) instancesRef.current[key].destroy();
  try {
    const theme = getChartTheme();
    instancesRef.current[key] = new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: (datasets || []).map((item) => ({
          ...item,
          borderWidth: 1.2,
          borderRadius: 8,
          borderSkipped: false,
          maxBarThickness: 26,
          categoryPercentage: 0.72
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        animation: { duration: 520, easing: "easeOutQuart" },
        scales: {
          x: {
            grid: { color: theme.grid, drawBorder: false },
            ticks: { color: theme.muted, maxRotation: 35, minRotation: 0 }
          },
          y: {
            grid: { color: theme.grid, drawBorder: false },
            ticks: { color: theme.muted },
            suggestedMin: 0,
            suggestedMax: 1
          }
        },
        plugins: {
          legend: {
            position: "top",
            labels: {
              color: theme.text,
              usePointStyle: true,
              pointStyle: "rectRounded",
              boxWidth: 10,
              boxHeight: 10,
              padding: 14
            }
          },
          tooltip: {
            backgroundColor: theme.tooltipBg,
            borderColor: theme.tooltipBorder,
            borderWidth: 1,
            titleColor: theme.text,
            bodyColor: theme.text,
            padding: 10
          }
        }
      }
    });
  } catch {}
}

function fillGaps(values) {
  if (!Array.isArray(values)) return [];
  const result = [...values];
  let last = null;
  for (let i = 0; i < result.length; i += 1) {
    const v = result[i];
    if (v == null || Number.isNaN(Number(v))) {
      if (last != null) result[i] = last;
      continue;
    }
    last = Number(v);
    result[i] = Number(v);
  }
  return result;
}

function renderAllCharts(instancesRef, refs, chartSeries, lossRange) {
  const epochsAll = chartSeries?.epochs || [];
  let start = 0;
  if (lossRange === "last30") start = Math.max(0, epochsAll.length - 30);
  if (lossRange === "last10") start = Math.max(0, epochsAll.length - 10);
  const epochs = epochsAll.slice(start);
  const trainLoss = chartSeries?.train_losses || {};
  const mapSeries = chartSeries?.map_series || {};
  const lrSeries = chartSeries?.lr_series || {};
  const distill = chartSeries?.distill_series || {};
  const pr = chartSeries?.pr_curve || {};
  const cls = chartSeries?.class_performance || {};
  renderLineChart(instancesRef, refs.lossRef.current, "loss", epochs, [
    { label: "Box Loss", data: (trainLoss.box_loss || []).slice(start), color: "#6750A4" },
    { label: "CLS Loss", data: (trainLoss.cls_loss || []).slice(start), color: "#F57C00" },
    { label: "DFL Loss", data: (trainLoss.dfl_loss || []).slice(start), color: "#1565C0" }
  ]);

  renderLineChart(instancesRef, refs.mapRef.current, "map", epochs, [
    { label: "mAP50", data: mapSeries.map50 || [], color: "#2E7D32" },
    { label: "mAP50-95", data: mapSeries.map50_95 || [], color: "#7D5260" },
    { label: "Precision", data: chartSeries?.precision_recall?.precision || [], color: "#F57C00" },
    { label: "Recall", data: chartSeries?.precision_recall?.recall || [], color: "#1565C0" }
  ]);

  renderLineChart(instancesRef, refs.lrRef.current, "lr", epochs, [
    { label: "LR pg0", data: lrSeries.pg0 || [], color: "#6750A4" },
    { label: "LR pg1", data: lrSeries.pg1 || [], color: "#625B71" },
    { label: "LR pg2", data: lrSeries.pg2 || [], color: "#7D5260" }
  ], { scales: { y: { type: "logarithmic" } } });

  renderLineChart(instancesRef, refs.distillRef.current, "distill", epochsAll, [
    { label: "Alpha", data: fillGaps(distill.alpha || []), color: "#6750A4" },
    { label: "Temperature T", data: fillGaps(distill.temperature || []), color: "#B3261E" },
    { label: "KD Loss", data: fillGaps(distill.kd_loss || []), color: "#F57C00" }
  ]);

  renderLineChart(instancesRef, refs.prRef.current, "pr", (pr.recall || []).map((r) => Number(r).toFixed(2)), [
    { label: "Precision-Recall", data: pr.precision || [], color: "#2E7D32" }
  ], { scales: { y: { min: 0, max: 1.05 } } });

  renderBarChart(instancesRef, refs.classRef.current, "class", cls.labels || ["Overall"], [
    { label: "mAP", data: cls.map || [0], backgroundColor: "#6750A4AA", borderColor: "#6750A4" },
    { label: "Recall", data: cls.recall || [0], backgroundColor: "#2E7D32AA", borderColor: "#2E7D32" },
    { label: "Precision", data: cls.precision || [0], backgroundColor: "#F57C00AA", borderColor: "#F57C00" }
  ]);
}

function AgentPanel({ toast }) {
  const [apiUrl, setApiUrl] = useState(() => window.localStorage.getItem("edge_distill_agent_api_url") || "");
  const [apiKey, setApiKey] = useState(() => window.localStorage.getItem("edge_distill_agent_api_key") || "");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [outputText, setOutputText] = useState("请先配置外部 Agent API，然后开始对话或调用动作。");
  const [approvalOpen, setApprovalOpen] = useState(false);
  const [approvalBody, setApprovalBody] = useState("");
  const [approvalToken, setApprovalToken] = useState("");

  const saveConfig = () => {
    window.localStorage.setItem("edge_distill_agent_api_url", apiUrl.trim());
    window.localStorage.setItem("edge_distill_agent_api_key", apiKey.trim());
    toast("Agent API 配置已保存", "success");
  };

  const send = async () => {
    const text = input.trim();
    if (!text) return;
    if (!apiUrl.trim()) {
      toast("请先填写 Agent API 地址", "warning");
      return;
    }
    setLoading(true);
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    try {
      const res = await fetch(apiUrl.trim(), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiKey.trim() ? { Authorization: apiKey.trim() } : {})
        },
        body: JSON.stringify({ query: text })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Agent 请求失败");
      }
      const reply = typeof data === "string" ? data : data.reply || data.message || JSON.stringify(data, null, 2);
      setMessages((prev) => [...prev, { role: "agent", content: reply }]);
      setOutputText(JSON.stringify(data, null, 2));
      await maybeHandlePatch(data, reply);
    } catch (error) {
      setMessages((prev) => [...prev, { role: "agent", content: `请求失败: ${error.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const maybeHandlePatch = async (result, replyText) => {
    const patch = extractPatchFromResult(result, replyText);
    if (!patch) return;
    try {
      const preview = await apiRequest("/api/agent/patch/preview", {
        method: "POST",
        body: JSON.stringify({ patch })
      });
      setApprovalBody(`${preview.patch_yaml || ""}\n\n--- merged_preview ---\n${JSON.stringify(preview.merged_preview || {}, null, 2)}`);
      setApprovalToken(preview.approval_token || "");
      setApprovalOpen(true);
      setOutputText(JSON.stringify(preview, null, 2));
      setMessages((prev) => [...prev, { role: "agent", content: "已生成配置 patch 预览，请确认后写入。" }]);
    } catch (error) {
      setMessages((prev) => [...prev, { role: "agent", content: `Patch 校验失败: ${error.message}` }]);
    }
  };

  const testAgentApi = async () => {
    if (!apiUrl.trim()) return toast("请先填写 Agent API 地址", "warning");
    try {
      const res = await fetch(apiUrl.trim(), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiKey.trim() ? { Authorization: apiKey.trim() } : {})
        },
        body: JSON.stringify({ action: "ping", params: {} })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      toast("Agent API 连接成功", "success");
    } catch (error) {
      toast(`Agent API 连接失败: ${error.message}`, "error");
    }
  };

  const loadSchema = async () => {
    try {
      const data = await apiRequest("/api/agent/config-schema");
      setOutputText(JSON.stringify(data, null, 2));
      toast("已加载配置结构", "success");
    } catch (error) {
      toast(error.message, "error");
    }
  };

  const parseClipboardPatch = async () => {
    try {
      const text = await navigator.clipboard.readText();
      const patch = extractPatchFromResult(null, text);
      if (!patch) return toast("剪贴板中未识别到 patch", "warning");
      await maybeHandlePatch({ patch }, "");
    } catch {
      toast("无法读取剪贴板（需浏览器权限）", "warning");
    }
  };

  const applyApproval = async () => {
    if (!approvalToken) return;
    try {
      const data = await apiRequest("/api/agent/patch/apply", {
        method: "POST",
        body: JSON.stringify({ approval_token: approvalToken })
      });
      setApprovalOpen(false);
      setApprovalToken("");
      setOutputText(JSON.stringify(data, null, 2));
      toast("已写入 distill_config.yaml", "success");
    } catch (error) {
      toast(error.message, "error");
    }
  };

  return (
    <div className="tab-panel active" id="panel-agent">
      <div className="agent-layout">
        <div className="agent-sidebar">
          <h3 className="sidebar-title"><span className="material-icons">hub</span>连接与工具</h3>
          <div className="agent-settings-card">
            <h4>外部 API 配置</h4>
            <div className="form-row stacked-row">
              <TextField label="Agent API 地址" value={apiUrl} onChange={setApiUrl} />
              <TextField label="API Token (可选)" value={apiKey} onChange={setApiKey} />
            </div>
            <div className="launch-actions" style={{ marginTop: 12 }}>
              <button className="md-btn md-btn-tonal" onClick={saveConfig}>
                <span className="material-icons">save</span>保存配置
              </button>
              <button className="md-btn md-btn-outlined" onClick={testAgentApi}>
                <span className="material-icons">bolt</span>测试连接
              </button>
            </div>
          </div>
          <div className="agent-local-tools">
            <h4 className="tools-title"><span className="material-icons">verified_user</span>本地审批</h4>
            <p className="tools-desc">外部 Agent 返回的 patch 仅预览；需批准后才写入配置。</p>
            <div className="tools-actions">
              <button type="button" className="md-btn md-btn-tonal md-btn-compact" onClick={loadSchema}>
                <span className="material-icons">schema</span>配置结构
              </button>
              <button type="button" className="md-btn md-btn-outlined md-btn-compact" onClick={parseClipboardPatch}>
                <span className="material-icons">content_paste</span>剪贴板 Patch
              </button>
            </div>
          </div>
        </div>
        <div className="agent-main">
          <div className="agent-chat-panel">
            <div className="chat-header">
              <h3><span className="material-icons">chat</span>对话</h3>
              <div className="chat-controls">
                <button className="btn-icon-sm" onClick={() => setMessages([])}><span className="material-icons">delete_sweep</span></button>
              </div>
            </div>
            <div id="agent-chat-messages" className="chat-messages">
              {messages.map((msg, index) => (
                <div key={`${msg.role}-${index}`} className={`chat-message ${msg.role}`}>
                  <div className={`message-avatar ${msg.role === "agent" ? "agent-avatar" : "user-avatar"}`}>
                    <span className="material-icons">{msg.role === "agent" ? "smart_toy" : "person"}</span>
                  </div>
                  <div className="message-content"><div>{msg.content}</div></div>
                </div>
              ))}
              {loading ? (
                <div className="chat-message agent">
                  <div className="message-avatar agent-avatar"><span className="material-icons">smart_toy</span></div>
                  <div className="message-content"><div>处理中...</div></div>
                </div>
              ) : null}
            </div>
            <div className="chat-input-area">
              <div className="chat-input-wrapper">
                <textarea
                  rows={2}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      send();
                    }
                  }}
                  placeholder="输入消息；外部 API 可返回 patch 字段..."
                />
                <button className="md-btn md-btn-filled primary send-btn" type="button" onClick={send} disabled={loading}>
                  <span className="material-icons">send</span>
                </button>
              </div>
            </div>
          </div>
          <div className="agent-status-panel">
            <div className="status-header"><h4>Agent 输出</h4></div>
            <div id="agent-output" className="agent-output">
              <pre id="agent-output-content">{outputText}</pre>
            </div>
          </div>
        </div>
      </div>
      {approvalOpen ? (
        <div className="md-dialog" style={{ display: "block" }}>
          <div className="md-dialog-surface">
            <h2 className="md-dialog-title">批准修改训练配置？</h2>
            <p className="md-dialog-support">将写入 configs/distill_config.yaml 并刷新训练配置表单。</p>
            <pre className="md-dialog-pre">{approvalBody}</pre>
            <div className="md-dialog-actions">
              <button type="button" className="md-btn md-btn-text" onClick={() => setApprovalOpen(false)}>取消</button>
              <button type="button" className="md-btn md-btn-filled primary" onClick={applyApproval}>
                <span className="material-icons">check_circle</span>批准并写入
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function extractPatchFromResult(result, text) {
  if (result && typeof result.patch === "object") return result.patch;
  if (result && typeof result.suggested_patch === "object") return result.suggested_patch;
  const raw = typeof text === "string" ? text.trim() : "";
  if (!raw) return null;
  const match = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = match ? match[1].trim() : raw;
  try {
    const parsed = JSON.parse(candidate);
    if (parsed && typeof parsed.patch === "object") return parsed.patch;
    if (parsed && (parsed.distillation || parsed.training || parsed.output)) return parsed;
  } catch {
    return null;
  }
  return null;
}

export default App;
