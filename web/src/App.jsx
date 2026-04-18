import { useEffect, useLayoutEffect, useRef, useState } from "react";
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
import { apiRequest } from "./api/client";
import { DEFAULT_FORM, COMPUTE_PRESETS, inferComputeProviderFromConfig } from "./constants/trainingDefaults";
import { useToast } from "./hooks/useToast";
import { detectLogLevel } from "./utils/logging";
import { formatTime } from "./utils/time";

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

function App() {
  const [activeTab, setActiveTab] = useState("training");
  const [metricsCsvPath, setMetricsCsvPath] = useState("");
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
          <span className="app-subtitle">边缘蒸馏训练工作台</span>
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
          <TrainingPanel toast={push} active={activeTab === "training"} />
          <MetricsPanel toast={push} active={activeTab === "metrics"} onMetricsSourceChange={setMetricsCsvPath} />
          <AgentPanel toast={push} active={activeTab === "agent"} metricsCsvPath={metricsCsvPath} />
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

function TrainingPanel({ toast, active }) {
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
  const outputNameInputRef = useRef(null);
  const pendingOverlapAlertRef = useRef(false);
  const isRemoteApi = form.training.compute_provider === "remote_api";
  const datasetSource = form.training?.dataset_api?.source || (form.training?.dataset_api?.enabled ? "api" : "path");
  const useDatasetApi = isRemoteApi && datasetSource === "api";

  const setNested = (scope, key, value) => {
    setForm((prev) => ({
      ...prev,
      [scope]: { ...prev[scope], [key]: value }
    }));
  };

  const updateTrainingNested = (section, patch) => {
    setForm((prev) => ({
      ...prev,
      training: {
        ...prev.training,
        [section]: { ...prev.training?.[section], ...patch }
      }
    }));
  };

  const normalizeWandbForUi = (wandb) => {
    const next = { ...wandb };
    if (Array.isArray(next.tags)) {
      next.tags = next.tags.map((t) => String(t).trim()).filter(Boolean).join(", ");
    }
    return next;
  };

  const buildConfigPayload = (sourceForm) => {
    const payload = JSON.parse(JSON.stringify(sourceForm || {}));
    const wandb = payload.wandb || {};
    const rawTags = wandb.tags;
    if (typeof rawTags === "string") {
      wandb.tags = rawTags
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    } else if (!Array.isArray(rawTags)) {
      wandb.tags = [];
    }
    payload.wandb = wandb;
    return payload;
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
    const inferredProvider = inferComputeProviderFromConfig(config, form.training?.compute_provider || "local");
    setForm((prev) => ({
      ...prev,
      ...config,
      distillation: { ...prev.distillation, ...config?.distillation },
      training: {
        ...prev.training,
        ...config?.training,
        cloud_api: { ...prev.training?.cloud_api, ...config?.training?.cloud_api },
        dataset_api: { ...prev.training?.dataset_api, ...config?.training?.dataset_api },
        compute_provider: inferredProvider
      },
      output: { ...prev.output, ...config?.output },
      wandb: { ...prev.wandb, ...normalizeWandbForUi(config?.wandb) }
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
    if (!isOutputPathOverlap) {
      pendingOverlapAlertRef.current = false;
      return;
    }
    if (!isOutputPathOverlap || !currentOutputName) return;
    const overlapKey = `${currentOutputProject}/${currentOutputName}`;
    if (overlapAlertShownRef.current === overlapKey) return;
    const isFocused = document.activeElement === outputNameInputRef.current;
    if (isFocused) {
      pendingOverlapAlertRef.current = true;
      return;
    }
    pendingOverlapAlertRef.current = false;
    overlapAlertShownRef.current = overlapKey;
    window.alert(`路径重合：${overlapKey}`);
    toast(`路径重合：${overlapKey}`, "warning");
  }, [currentOutputProject, currentOutputName, isOutputPathOverlap]);

  const saveConfig = async () => {
    await apiRequest("/api/config/save", {
      method: "POST",
      body: JSON.stringify({ name: "distill_config.yaml", config: buildConfigPayload(form) })
    });
  };

  const applyComputePreset = (provider) => {
    const preset = COMPUTE_PRESETS[provider] || COMPUTE_PRESETS.local;
    setForm((prev) => ({
      ...prev,
      training: {
        ...prev.training,
        compute_provider: provider,
        device: String(prev.training?.device || "").trim() ? prev.training.device : preset.device
      },
      output: {
        ...prev.output,
        project: preset.outputProject
      }
    }));
    refreshRunNameSuggestion(preset.outputProject, form.output.name, true);
    refreshResumeCandidates(preset.outputProject, false);
  };

  const startTraining = async () => {
    const studentWeight = form.distillation.student_weight?.trim();
    const teacherWeight = form.distillation.teacher_weight?.trim();
    const dataYaml = form.training.data_yaml?.trim();
    if (!studentWeight) return toast("请选择学生模型权重文件", "warning");
    if (!useDatasetApi && !dataYaml) return toast("请填写数据集配置文件路径", "warning");
    if (useDatasetApi && !form.training?.dataset_api?.resolve_url?.trim()) {
      return toast("已选择数据集 API，请填写数据集 API URL", "warning");
    }
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
    <div className={`tab-panel ${active ? "active" : ""}`} id="panel-training" aria-hidden={!active}>
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
                <PathField
                  label="项目目录"
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
                    disabled={running}
                  />
                  <label className="md-field-label">运行名称</label>
                </div>
                <small className={`hint ${isOutputPathOverlap ? "warning" : ""}`}>{renderedHint || runHint}</small>
              </div>
            </div>
          </div>
          <div className={`config-card launcher-side-card ${mode !== "resume" ? "disabled-panel" : ""}`}>
            <h3 className="card-header">续训历史</h3>
            <div className="form-group">
              <label>请选择历史运行</label>
              <M3Select
                value={String(selectedResumeIndex)}
                onChange={(nextValue) => {
                  const idx = Number(nextValue) || 0;
                  setSelectedResumeIndex(idx);
                  const c = resumeCandidates[idx];
                  if (c) {
                    setForm((prev) => ({
                      ...prev,
                      output: { ...prev.output, project: c.project, name: c.name }
                    }));
                  }
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
                disabled={running || useDatasetApi}
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
                disabled={running}
              />
              {isRemoteApi ? (
                <>
                  <SelectField
                    label="数据集来源"
                    value={datasetSource}
                    onChange={(v) => updateTrainingNested("dataset_api", { source: v, enabled: v === "api" })}
                    options={[
                      { value: "path", label: "本地/YAML 路径" },
                      { value: "api", label: "数据集 API" }
                    ]}
                    disabled={running}
                  />
                  <TextField
                    label="云训练 API Base URL"
                    value={form.training?.cloud_api?.base_url || ""}
                    onChange={(v) => updateTrainingNested("cloud_api", { base_url: v })}
                    disabled={running}
                  />
                  <TextField
                    label="云训练 API Token (可选)"
                    value={form.training?.cloud_api?.token || ""}
                    onChange={(v) => updateTrainingNested("cloud_api", { token: v })}
                    disabled={running}
                  />
                  {useDatasetApi ? (
                    <>
                      <TextField
                        label="数据集 API URL"
                        value={form.training?.dataset_api?.resolve_url || ""}
                        onChange={(v) => updateTrainingNested("dataset_api", { resolve_url: v })}
                        disabled={running}
                      />
                      <TextField
                        label="数据集 API Token (可选，默认复用云训练 Token)"
                        value={form.training?.dataset_api?.token || ""}
                        onChange={(v) => updateTrainingNested("dataset_api", { token: v })}
                        disabled={running}
                      />
                      <TextField
                        label="数据集名称/别名 (可选)"
                        value={form.training?.dataset_api?.dataset_name || ""}
                        onChange={(v) => updateTrainingNested("dataset_api", { dataset_name: v })}
                        disabled={running}
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
                  disabled={running}
                />
              ) : null}
              <NumberField label="训练轮数 Epochs" value={form.training.epochs} step="1" onChange={(v) => setNested("training", "epochs", v)} disabled={running} />
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
                disabled={running}
              />
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

          <div className="config-card">
            <h3 className="card-header">W&B 配置</h3>
            <div className="form-grid">
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
  const hasValue = value !== undefined && value !== null && String(value) !== "";
  return (
    <div className="form-group">
      <div className={`md-field ${hasValue ? "has-value" : ""}`}>
        <input className="md-input" value={value || ""} onChange={(e) => onChange(e.target.value)} disabled={disabled} placeholder=" " />
        <label className="md-field-label">{label}</label>
      </div>
    </div>
  );
}

function PathField({ label, value, onChange, onBrowse, disabled }) {
  const hasValue = value !== undefined && value !== null && String(value) !== "";
  return (
    <div className="form-group">
      <div className={`file-input-wrapper md-field ${hasValue ? "has-value" : ""}`}>
        <input className="md-input" value={value || ""} onChange={(e) => onChange(e.target.value)} disabled={disabled} placeholder=" " />
        {label ? <label className="md-field-label">{label}</label> : null}
        <button type="button" className="file-picker-label" onClick={onBrowse} disabled={disabled} title="浏览本地路径">
          <span className="material-icons">folder_open</span>
        </button>
      </div>
    </div>
  );
}

function NumberField({ label, value, onChange, step, disabled }) {
  const hasValue = value !== undefined && value !== null && String(value) !== "";
  return (
    <div className="form-group">
      <div className={`md-field ${hasValue ? "has-value" : ""}`}>
        <input className="md-input" type="number" step={step} value={value ?? ""} onChange={(e) => onChange(Number(e.target.value))} disabled={disabled} placeholder=" " />
        <label className="md-field-label">{label}</label>
      </div>
    </div>
  );
}

function M3Select({ value, onChange, options, disabled, className = "", ariaLabel = "选择项" }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const normalizedValue = value == null ? "" : String(value);
  const normalizedOptions = Array.isArray(options) ? options.map((opt) => ({ ...opt, value: String(opt.value) })) : [];
  const selected = normalizedOptions.find((opt) => opt.value === normalizedValue) || normalizedOptions[0];

  useEffect(() => {
    if (!open) return undefined;
    const onDocPointerDown = (event) => {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(event.target)) {
        setOpen(false);
      }
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onDocPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onDocPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className={`m3-select ${open ? "open" : ""} ${disabled ? "disabled" : ""} ${className}`.trim()}>
      <button
        type="button"
        className="m3-select-trigger md-input"
        onClick={() => {
          if (disabled) return;
          setOpen((prev) => !prev);
        }}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={ariaLabel}
        disabled={disabled}
      >
        <span className="m3-select-value">{selected?.label || ""}</span>
        <span className="material-icons m3-select-arrow">expand_more</span>
      </button>
      {open ? (
        <div className="m3-select-menu" role="listbox">
          {normalizedOptions.map((opt) => (
            <button
              key={opt.value}
              type="button"
              role="option"
              aria-selected={opt.value === normalizedValue}
              className={`m3-select-option ${opt.value === normalizedValue ? "selected" : ""}`}
              onClick={() => {
                onChange(opt.value);
                setOpen(false);
              }}
            >
              <span>{opt.label}</span>
              {opt.value === normalizedValue ? <span className="material-icons">check</span> : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function SelectField({ label, value, onChange, options, disabled }) {
  const hasValue = value !== undefined && value !== null && String(value) !== "";
  return (
    <div className="form-group">
      <div className={`md-field md-field-select ${hasValue ? "has-value" : ""}`}>
        <M3Select value={value} onChange={onChange} options={options} disabled={disabled} ariaLabel={label} />
        <label className="md-field-label">{label}</label>
      </div>
    </div>
  );
}

function MetricsPanel({ toast, active, onMetricsSourceChange }) {
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
    if (typeof onMetricsSourceChange === "function") {
      onMetricsSourceChange(source || "");
    }
  }, [source, onMetricsSourceChange]);

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
    <div className={`tab-panel ${active ? "active" : ""}`} id="panel-metrics" aria-hidden={!active}>
      <div className="metrics-toolbar">
        <div className="toolbar-left">
          <button className="md-btn md-btn-outlined metrics-refresh-btn" onClick={() => refreshSources(true)}>
            <span className="material-icons">refresh</span>刷新数据
          </button>
          <M3Select
            className="metrics-source-select"
            value={source}
            onChange={(next) => setSource(next)}
            options={
              sources.length === 0
                ? [{ value: "", label: "暂无训练结果可选" }]
                : sources.map((item) => ({ value: item.path, label: item.display_name || item.name }))
            }
            ariaLabel="选择指标来源"
          />
        </div>
        <div className="toolbar-right metrics-toolbar-right">
          <div className="chip-group">
            <button className={`chip ${chartType === "loss" ? "active" : ""}`} onClick={() => setChartType("loss")}>损失</button>
            <button className={`chip ${chartType === "accuracy" ? "active" : ""}`} onClick={() => setChartType("accuracy")}>精度</button>
            <button className={`chip ${chartType === "lr" ? "active" : ""}`} onClick={() => setChartType("lr")}>学习率</button>
            <button className={`chip ${chartType === "all" ? "active" : ""}`} onClick={() => setChartType("all")}>全部</button>
          </div>
          <div className="auto-refresh-bar">
            <label className="auto-refresh-label">
              <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
              自动刷新图表数据
            </label>
            <span className="auto-refresh-interval">{autoRefresh ? `${refreshLeft}s 后刷新` : "--"}</span>
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
                  <M3Select
                    className="mini-select"
                    value={lossRange}
                    onChange={(next) => setLossRange(next)}
                    options={[
                      { value: "all", label: "全部 Epochs" },
                      { value: "last30", label: "最近 30" },
                      { value: "last10", label: "最近 10" }
                    ]}
                    ariaLabel="选择损失范围"
                  />
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

function formatMetricValueForTooltip(chartKey, value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "--";
  if (chartKey === "lr") {
    if (num === 0) return "0";
    if (Math.abs(num) < 0.001) return num.toExponential(6);
    return num.toFixed(6);
  }
  return num.toFixed(4);
}

function compactClassSeries(cls = {}) {
  const labels = Array.isArray(cls.labels) ? cls.labels : [];
  const map = Array.isArray(cls.map) ? cls.map : [];
  const recall = Array.isArray(cls.recall) ? cls.recall : [];
  const precision = Array.isArray(cls.precision) ? cls.precision : [];
  const toMetric = (arr, idx) => {
    const raw = arr[idx];
    if (raw == null || raw === "") return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  };
  const next = { labels: [], map: [], recall: [], precision: [] };
  for (let i = 0; i < labels.length; i += 1) {
    const m = toMetric(map, i);
    const r = toMetric(recall, i);
    const p = toMetric(precision, i);
    if (m == null && r == null && p == null) continue;
    next.labels.push(labels[i]);
    next.map.push(m);
    next.recall.push(r);
    next.precision.push(p);
  }
  return next;
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
            padding: 10,
            callbacks: {
              label: (ctx) => {
                const chartKey = key;
                const value = chartKey === "lr" ? ctx?.raw : (ctx?.parsed?.y ?? ctx?.raw);
                return `${ctx.dataset.label}: ${formatMetricValueForTooltip(chartKey, value)}`;
              }
            }
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
            padding: 10,
            callbacks: {
              label: (ctx) => {
                const value = ctx?.parsed?.y ?? ctx?.raw;
                return `${ctx.dataset.label}: ${formatMetricValueForTooltip(key, value)}`;
              }
            }
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
  const cls = compactClassSeries(chartSeries?.class_performance || {});
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

/** 侧边栏「常用工具」：点击即作为用户消息发送（高频问句） */
const AGENT_QUICK_PROMPTS = [
  {
    id: "eval_latest",
    label: "评价训练结果",
    text: "请结合当前训练指标与输出目录，评价最近一次训练结果，并指出主要问题与改进方向；如需改配置，请用 Markdown 代码块给出可在 PowerShell 中执行的命令。"
  },
  {
    id: "analyze_params",
    label: "分析蒸馏参数",
    text: "请先按约定输出 tool JSON 调用 agent.get_context，再调用 agent.analyze_params 获取事实，最后给出优化建议；涉及修改 configs/distill_config.yaml 时，请用 Markdown 代码块输出 PowerShell 命令（如 notepad、code、或 Python 一行读写），不要只给泛泛而谈。"
  },
  {
    id: "list_runs",
    label: "实验目录",
    text: "请使用工具说明当前训练输出目录、exp 命名规则，以及如何定位某次实验的权重与曲线。"
  },
  {
    id: "metrics_report",
    label: "指标摘要",
    text: "请根据当前训练指标生成简短摘要，并给出下一步建议；若需要本地命令，请用代码块给出。"
  }
];

/** 从模型回复中提取「可执行」代码块（排除单行 tool JSON） */
function extractExecutableFences(reply) {
  const raw = typeof reply === "string" ? reply : "";
  const out = [];
  const fenceRe = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let m;
  while ((m = fenceRe.exec(raw)) !== null) {
    const inner = (m[2] || "").trim();
    if (!inner) continue;
    if (/^\s*\{\s*"tool"\s*:/.test(inner) && inner.length < 1200) continue;
    const lang = (m[1] || "").trim().toLowerCase();
    if (lang === "json" && /"tool"\s*:/.test(inner)) continue;
    out.push(inner);
  }
  return out;
}

const _TERMINAL_TOOL_JSON_MAX = 14000;

/** 右侧「Agent 输出」：分节展示，避免与对话重复堆叠 */
function formatAgentTerminalOutput(reply, toolLogs) {
  const fences = extractExecutableFences(reply);
  const heuristicLines = (typeof reply === "string" ? reply : "")
    .split("\n")
    .filter((line) => {
      const t = line.trim();
      if (!t || t.startsWith("```")) return false;
      return /^(python|pip|cd|git|curl|notepad|code|\$|\.\/|Invoke-)/i.test(t);
    })
    .join("\n")
    .trim();

  const sections = [];

  if (fences.length) {
    sections.push("【可执行命令 / 脚本】", fences.join("\n\n────────\n\n"));
  } else if (heuristicLines) {
    sections.push("【疑似命令行（从正文提取）】", heuristicLines);
  }

  if (toolLogs && toolLogs.length) {
    sections.push("");
    sections.push("【本地工具调用】");
    toolLogs.forEach((t, i) => {
      const name = t.call?.tool || "?";
      const args = t.call?.args && typeof t.call.args === "object" ? t.call.args : {};
      sections.push(`── ${i + 1}. ${name} ──`);
      sections.push(`请求参数:\n${JSON.stringify({ tool: name, args }, null, 2)}`);
      let resStr = "";
      try {
        resStr = JSON.stringify(t.result, null, 2);
      } catch {
        resStr = String(t.result);
      }
      const fullLen = resStr.length;
      if (fullLen > _TERMINAL_TOOL_JSON_MAX) {
        resStr = `${resStr.slice(0, _TERMINAL_TOOL_JSON_MAX)}\n… (以下省略 ${fullLen - _TERMINAL_TOOL_JSON_MAX} 字符)`;
      }
      sections.push(`返回:\n${resStr}`);
      sections.push("");
    });
  }

  if (!sections.length) {
    return "【提示】本轮未识别到可单独摘出的终端代码块，且未经过本地工具。\n完整说明见左侧对话；需要命令时请让模型用 ```powershell``` 或 ```bash``` 代码块输出。";
  }

  if (!fences.length && !heuristicLines && toolLogs?.length) {
    sections.unshift("【提示】模型未使用 Markdown 代码块给出 shell；下方为工具原始返回。");
  }

  return sections.join("\n").trim();
}

/** 从 OpenAI 风格 message.content 提取文本（多模态为数组时拼接） */
function pieceTextFromContent(val) {
  if (val == null) return "";
  if (typeof val === "string") return val;
  if (Array.isArray(val)) {
    const parts = [];
    for (const item of val) {
      if (typeof item === "string") parts.push(item);
      else if (item && typeof item === "object") {
        if (typeof item.text === "string") parts.push(item.text);
        else if (typeof item.content === "string") parts.push(item.content);
      }
    }
    return parts.join("");
  }
  return String(val);
}

/** 从 API 响应中同时取出正文与 reasoning（与后端 _extract_openai_reasoning_from_message 对齐） */
function extractReplyAndReasoningFromPayload(payload) {
  if (typeof payload === "string") return { reply: payload, reasoning: "" };
  if (!payload || typeof payload !== "object") return { reply: String(payload ?? ""), reasoning: "" };
  let reasoning = "";
  if (typeof payload.reasoning === "string" && payload.reasoning.trim()) {
    reasoning = payload.reasoning.trim();
  }
  const msg = payload?.choices?.[0]?.message;
  if (msg && typeof msg === "object") {
    if (!reasoning) {
      const r = msg.reasoning_content ?? msg.reasoning;
      if (typeof r === "string" && r.trim()) reasoning = r.trim();
    }
    const reply = pieceTextFromContent(msg.content);
    if (reply.trim()) return { reply, reasoning };
  }
  const fallback = payload.reply ?? payload.message ?? payload.output;
  if (typeof fallback === "string") return { reply: fallback, reasoning };
  if (fallback !== undefined && fallback !== null) {
    return { reply: typeof fallback === "object" ? JSON.stringify(fallback, null, 2) : String(fallback), reasoning };
  }
  return { reply: JSON.stringify(payload, null, 2), reasoning };
}

/** 从正文中剥离常见「思考」包裹块（不进入 UI「思考过程」区，仅清洁正文） */
function splitEmbeddedReasoningFromReply(text) {
  const raw = typeof text === "string" ? text : "";
  let main = raw;
  const patterns = [
    /<think\b[^>]*>([\s\S]*?)<\/think>/gi,
    /<think\b[^>]*>([\s\S]*?)<\/redacted_thinking>/gi,
    /<redacted_thinking>([\s\S]*?)<\/redacted_thinking>/gi
  ];
  for (const re of patterns) {
    main = main.replace(re, () => "");
  }
  main = main.replace(/\n{3,}/g, "\n\n").trim();
  return { main };
}

/** displayReply 始终剥离内嵌块；displayReasoning 仅来自 API */
function buildDisplayReplyAndReasoning(rawReply, reasoningFromApi) {
  const { main } = splitEmbeddedReasoningFromReply(rawReply);
  const apiReason =
    typeof reasoningFromApi === "string" && reasoningFromApi.trim() ? reasoningFromApi.trim() : "";
  return { displayReply: main, displayReasoning: apiReason };
}

/** Agent 气泡：工具 chips、仅 API 思考可折叠、``` 代码块排版 */
function ChatBubbleBody({ role, content, reasoningApi, toolsUsed, streaming }) {
  const text = content ?? "";
  const reasoningText = typeof reasoningApi === "string" && reasoningApi.trim() ? reasoningApi.trim() : "";
  const tools = Array.isArray(toolsUsed) ? toolsUsed.filter(Boolean) : [];
  if (role !== "agent") {
    return <div className="chat-plain chat-pre-wrap">{text}</div>;
  }
  const parts = [];
  const fenceRe = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let idx = 0;
  let m;
  while ((m = fenceRe.exec(text)) !== null) {
    if (m.index > idx) {
      parts.push({ kind: "text", value: text.slice(idx, m.index) });
    }
    parts.push({ kind: "code", value: (m[2] || "").replace(/\n$/, "") });
    idx = m.index + m[0].length;
  }
  if (idx < text.length) {
    parts.push({ kind: "text", value: text.slice(idx) });
  }
  if (parts.length === 0) {
    parts.push({ kind: "text", value: text });
  }
  const answerBlock = (
    <>
      {parts.map((p, i) =>
        p.kind === "code" ? (
          <pre key={i} className="agent-inline-codefence">
            {p.value}
          </pre>
        ) : (
          <div key={i} className="agent-inline-text">
            {p.value}
          </div>
        )
      )}
    </>
  );

  return (
    <div className="agent-bubble-formatted">
      {tools.length > 0 ? (
        <div className="agent-bubble-tools" role="list" aria-label="已使用工具">
          {tools.map((name) => (
            <span key={name} className="md-chip md-chip-assist" role="listitem">
              <span className="material-icons md-chip-icon" aria-hidden>
                build
              </span>
              {name}
            </span>
          ))}
        </div>
      ) : null}
      {reasoningText ? (
        <details className="agent-reasoning-details" open={!!streaming}>
          <summary className="agent-reasoning-summary">
            思考过程
            {streaming ? <span className="agent-reasoning-live">生成中…</span> : null}
          </summary>
          <pre className="agent-reasoning-pre">{reasoningText}</pre>
        </details>
      ) : null}
      {reasoningText ? <div className="agent-answer-body">{answerBlock}</div> : answerBlock}
    </div>
  );
}

function AgentPanel({ toast, active, metricsCsvPath }) {
  const [apiUrl, setApiUrl] = useState(() => window.localStorage.getItem("edge_distill_agent_api_url") || "");
  const [apiKey, setApiKey] = useState(() => window.localStorage.getItem("edge_distill_agent_api_key") || "");
  const [apiModel, setApiModel] = useState(() => window.localStorage.getItem("edge_distill_agent_api_model") || "");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [outputText, setOutputText] = useState("请先配置外部 Agent API，然后开始对话或调用动作。");
  const [approvalOpen, setApprovalOpen] = useState(false);
  const [approvalBody, setApprovalBody] = useState("");
  const [approvalToken, setApprovalToken] = useState("");
  const [approvalRunId, setApprovalRunId] = useState("default");
  const [approvalRequestHash, setApprovalRequestHash] = useState("");
  const chatTextareaRef = useRef(null);
  const chatMessagesRef = useRef(null);
  const agentSlotIndexRef = useRef(-1);

  useLayoutEffect(() => {
    const el = chatMessagesRef.current;
    if (!el || !active) return;
    const applyScroll = () => {
      el.scrollTop = el.scrollHeight;
    };
    applyScroll();
    window.requestAnimationFrame(applyScroll);
  }, [messages, loading, active]);

  useEffect(() => {
    const el = chatMessagesRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      if (!active) return;
      el.scrollTop = el.scrollHeight;
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [active]);

  const resizeChatInput = () => {
    const el = chatTextareaRef.current;
    if (!el) return;
    const computed = window.getComputedStyle(el);
    const lineHeight = parseFloat(computed.lineHeight) || 20;
    const paddingTop = parseFloat(computed.paddingTop) || 0;
    const paddingBottom = parseFloat(computed.paddingBottom) || 0;
    const borderTop = parseFloat(computed.borderTopWidth) || 0;
    const borderBottom = parseFloat(computed.borderBottomWidth) || 0;
    const maxHeight = lineHeight * 3 + paddingTop + paddingBottom + borderTop + borderBottom;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
    el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
  };

  useEffect(() => {
    const el = chatTextareaRef.current;
    if (!el) return;
    resizeChatInput();
  }, [input]);

  const saveConfig = () => {
    window.localStorage.setItem("edge_distill_agent_api_url", apiUrl.trim());
    window.localStorage.setItem("edge_distill_agent_api_key", apiKey.trim());
    window.localStorage.setItem("edge_distill_agent_api_model", apiModel.trim());
    toast("Agent API 配置已保存", "success");
  };

  const resolveModelName = () => {
    const value = apiModel.trim();
    return value || "gpt-4o-mini";
  };

  const isArkApiUrl = () => {
    const text = apiUrl.trim().toLowerCase();
    return text.includes("ark.") && text.includes("/api/v");
  };

  const parseResponsePayload = async (res) => {
    const text = await res.text();
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  };

  const buildAuthHeaderCandidates = () => {
    const token = apiKey.trim();
    if (!token) return [{}];
    const withBearer = /^bearer\s+/i.test(token) ? token : `Bearer ${token}`;
    const variants = [
      { Authorization: token },
      { Authorization: withBearer },
      { "x-api-key": token },
      { Authorization: withBearer, "x-api-key": token }
    ];
    const seen = new Set();
    return variants.filter((item) => {
      const key = JSON.stringify(item);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  };

  const buildAgentTargets = () => {
    const raw = apiUrl.trim().replace(/\/+$/, "");
    const openaiPath = "/v1/chat/completions";
    const targets = [];
    const hasOpenAiEndpoint = /\/v1\/chat\/completions$/i.test(raw);
    if (hasOpenAiEndpoint) {
      targets.push({ kind: "openai", url: raw });
      return targets;
    }
    targets.push({ kind: "custom", url: raw });
    if (/\/v1$/i.test(raw)) {
      targets.push({ kind: "openai", url: `${raw}/chat/completions` });
    } else {
      targets.push({ kind: "openai", url: `${raw}${openaiPath}` });
    }
    return targets;
  };

  const extractToolCallFromText = (text) => {
    const raw = typeof text === "string" ? text.trim() : "";
    const toToolCall = (parsed) => {
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
      if (typeof parsed.tool === "string") {
        return { tool: parsed.tool, args: parsed.args && typeof parsed.args === "object" ? parsed.args : {} };
      }
      if (parsed.action === "tool_call" && typeof parsed.name === "string") {
        return { tool: parsed.name, args: parsed.arguments && typeof parsed.arguments === "object" ? parsed.arguments : {} };
      }
      return null;
    };

    const candidates = [];
    if (raw) candidates.push({ source: "raw", text: raw });
    const blocks = [...raw.matchAll(/```[a-zA-Z0-9_-]*\s*([\s\S]*?)```/g)];
    for (const m of blocks) {
      const body = (m[1] || "").trim();
      if (body) candidates.push({ source: "fence", text: body });
    }


    for (const c of candidates) {
      try {
        const parsed = JSON.parse(c.text);
        const tc = toToolCall(parsed);
        if (tc) {
          return tc;
        }
      } catch {
        /* continue */
      }
    }

    // 再做一次松散扫描：从混合文本中提取第一个可解析且符合 tool schema 的 JSON 对象
    for (let i = 0; i < raw.length; i += 1) {
      if (raw[i] !== "{") continue;
      let depth = 0;
      for (let j = i; j < raw.length; j += 1) {
        const ch = raw[j];
        if (ch === "{") depth += 1;
        else if (ch === "}") {
          depth -= 1;
          if (depth === 0) {
            const slice = raw.slice(i, j + 1);
            try {
              const parsed = JSON.parse(slice);
              const tc = toToolCall(parsed);
              if (tc) {
                return tc;
              }
            } catch {
              /* try next slice */
            }
            break;
          }
        }
      }
    }

    return null;
  };

  const buildAgentSystemPrompt = async () => {
    let contract = null;
    try {
      contract = await apiRequest("/api/agent/tools");
    } catch {
      contract = null;
    }
    const toolList = Array.isArray(contract?.tools)
      ? contract.tools.map((t) => `- ${t.name}: input=${JSON.stringify(t.input || {})}, output=${t.output || ""}`).join("\n")
      : "- agent.get_context\n- agent.analyze_params\n- agent.propose_patch\n- agent.validate_patch\n- agent.preview_patch\n- agent.apply_patch_with_approval\n- agent.list_run_history\n- agent.rollback_run_config";
    return [
      "你是训练参数优化 Agent。你可以且应该使用工具先获取事实，再给结论。",
      "可用工具如下：",
      toolList,
      "",
      "当需要调用工具时，你必须只输出一个 JSON 对象（不要输出其他文本）：",
      '{"tool":"agent.get_context","args":{"run_id":"default"}}',
      "工具结果会在下一轮以 tool 消息回传给你。",
      "**修改 distill 配置（configs/distill_config.yaml）时**：优先输出 `{\"tool\":\"agent.preview_patch\",\"args\":{\"patch\":{...}}}`（可先 `agent.validate_patch`）；若只调用了 `agent.propose_patch`，界面也会自动用返回的 patch 请求预览并在左侧栏显示「批准修改训练配置」面板。**禁止**在最终答复里用「是否需要我执行/是否生成补丁」等话术向用户索要确认。",
      "终端里的训练命令（```powershell``` / ```bash```）只能作为补充说明；真正写入配置必须经过上述工具链或界面审批。",
      "当不再需要工具、输出最终答复时：用自然语言说明变更理由与风险；若已调用 `agent.preview_patch`，只需提示用户在左侧栏「批准修改训练配置」面板中批准，不要重复询问。仅在确定无法走工具链时，才用 JSON 代码块给出结构化 patch 作为兜底。",
      "**审批/写入流程结束后**：若用户未主动要求继续改配置，你**不得**主动提出新的修改建议、不得追问「要不要继续调整」「是否还要改某参数」「需不需要再预览一版」等；**不得**自动再发起 `agent.propose_patch` / `agent.preview_patch` 或引导用户进入下一轮审批。此时只做简短收尾（例如已写入/已按侧栏操作即可训练），然后停止。",
      "**仅当用户明确说出**要改配置、改某字段、再优化、再出一版 patch 等意图时，你才可以再次使用配置相关工具或给出修改建议。"
    ].join("\n");
  };

  const inferRelayEndpointCandidates = (baseUrl) => {
    const base = String(baseUrl || "").trim().replace(/\/+$/, "");
    if (!base) return [null];
    if (/\/chat\/completions$/i.test(base) || /\/responses$/i.test(base) || /\/messages$/i.test(base)) {
      return [base, null];
    }
    const list = [];
    if (/\/api\/v\d+$/i.test(base)) {
      list.push(`${base}/chat/completions`);
    }
    list.push(`${base}/v1/chat/completions`);
    list.push(`${base}/chat/completions`);
    list.push(null);
    return Array.from(new Set(list));
  };

  /** 通过本地 HTTPS 中继读取 SSE：正文与思考分流式增量 */
  const readAgentInvokeSseStream = async (response, onDelta) => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let reply = "";
    let reasoning = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let lineEnd;
      while ((lineEnd = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, lineEnd);
        buffer = buffer.slice(lineEnd + 1);
        const trimmed = line.replace(/\r$/, "").trim();
        if (!trimmed.startsWith("data:")) continue;
        const dataStr = trimmed.slice(5).trim();
        if (!dataStr) continue;
        let ev;
        try {
          ev = JSON.parse(dataStr);
        } catch {
          continue;
        }
        if (ev.t === "content" && ev.d) {
          reply += ev.d;
          onDelta?.({ reply, reasoning });
        } else if (ev.t === "reasoning" && ev.d) {
          reasoning += ev.d;
          onDelta?.({ reply, reasoning });
        } else if (ev.t === "done") {
          if (typeof ev.reply === "string") reply = ev.reply;
          if (typeof ev.reasoning === "string") reasoning = ev.reasoning;
          onDelta?.({ reply, reasoning });
        } else if (ev.t === "error") {
          throw new Error(ev.message || "流式调用失败");
        }
      }
    }
    if (buffer.trim()) {
      const trimmed = buffer.replace(/\r$/, "").trim();
      if (trimmed.startsWith("data:")) {
        const dataStr = trimmed.slice(5).trim();
        try {
          const ev = JSON.parse(dataStr);
          if (ev.t === "done") {
            if (typeof ev.reply === "string") reply = ev.reply;
            if (typeof ev.reasoning === "string") reasoning = ev.reasoning;
            onDelta?.({ reply, reasoning });
          }
        } catch {
          /* ignore */
        }
      }
    }
    return { reply, reasoning };
  };

  const streamInvokeViaRelay = async ({ text, mode, systemPrompt, onDelta }) => {
    const base = apiUrl.trim();
    const modelName = resolveModelName();
    if (isArkApiUrl() && !apiModel.trim()) {
      throw new Error("检测到方舟地址，请先填写“模型名 / Endpoint ID”（如 ep-xxxxxx）");
    }
    const endpointCandidates = inferRelayEndpointCandidates(base);
    const errors = [];
    for (const endpoint of endpointCandidates) {
      try {
        const res = await fetch("/api/agent/model/invoke-stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            provider: "openai_compatible",
            api_url: base,
            api_key: apiKey.trim() || null,
            endpoint,
            model: modelName,
            temperature: 0.2,
            max_tokens: mode === "test" ? 8 : null,
            system_prompt: systemPrompt || null,
            messages: [{ role: "user", content: mode === "test" ? "ping" : text }]
          })
        });
        if (!res.ok) {
          let msg = `HTTP ${res.status}`;
          try {
            const errBody = await res.json();
            msg = errBody.error || errBody.message || msg;
          } catch {
            try {
              msg = (await res.text()) || msg;
            } catch {
              /* ignore */
            }
          }
          throw new Error(msg);
        }
        const { reply, reasoning } = await readAgentInvokeSseStream(res, onDelta);
        const payload = {
          status: "ok",
          reply,
          reasoning: reasoning || undefined
        };
        return { payload, target: { kind: "backend-relay", url: endpoint || base } };
      } catch (error) {
        let msg = error.message;
        if (!apiModel.trim() && /NotFound|does not exist|InvalidEndpointOrModel/i.test(String(msg || ""))) {
          msg = `${msg}（请在“模型名 / Endpoint ID”填写可用模型，如方舟控制台中的 endpoint-id）`;
        }
        errors.push(`relay@${endpoint || base}: ${msg}`);
      }
    }
    throw new Error(errors.slice(0, 3).join(" | ") || "本地中转流式调用失败");
  };

  const requestAgentWithFallback = async ({ text, mode, systemPrompt, onDelta, onRelayFallback }) => {
    const targets = buildAgentTargets();
    const authHeaders = buildAuthHeaderCandidates();
    const modelName = resolveModelName();
    if (isArkApiUrl() && !apiModel.trim()) {
      throw new Error("检测到方舟地址，请先填写“模型名 / Endpoint ID”（如 ep-xxxxxx）");
    }
    const errors = [];
    const prefersRelay = /^https?:\/\//i.test(apiUrl.trim());
    if (prefersRelay) {
      try {
        return await streamInvokeViaRelay({ text, mode, systemPrompt, onDelta });
      } catch (error) {
        errors.push(error.message);
        onRelayFallback?.();
      }
    }
    for (const target of targets) {
      for (const auth of authHeaders) {
        let body;
        if (target.kind === "openai") {
          const baseMessages = mode === "test" ? [{ role: "user", content: "ping" }] : [{ role: "user", content: text }];
          const withSystem = systemPrompt ? [{ role: "system", content: systemPrompt }, ...baseMessages] : baseMessages;
          body = mode === "test"
            ? { model: modelName, messages: withSystem, max_tokens: 1 }
            : { model: modelName, messages: withSystem, temperature: 0.2 };
        } else {
          body = mode === "test" ? { action: "ping", params: {} } : { query: text };
        }
        try {
          const res = await fetch(target.url, {
            method: "POST",
            headers: { "Content-Type": "application/json", ...auth },
            body: JSON.stringify(body)
          });
          const payload = await parseResponsePayload(res);
          if (!res.ok) {
            const message = typeof payload === "object" && payload ? payload.error || payload.message : String(payload || "");
            throw new Error(message || `HTTP ${res.status}`);
          }
          return { payload, target };
        } catch (error) {
          let msg = error.message;
          if (!apiModel.trim() && /NotFound|does not exist|InvalidEndpointOrModel/i.test(String(msg || ""))) {
            msg = `${msg}（请在“模型名 / Endpoint ID”填写可用模型）`;
          }
          errors.push(`${target.kind}@${target.url}: ${msg}`);
        }
      }
    }
    throw new Error(errors.slice(0, 3).join(" | ") || "所有连接方式均失败");
  };

  const runAgentWithTools = async ({ userText, maxRounds = 6 }) => {
    agentSlotIndexRef.current = -1;
    const systemPrompt = await buildAgentSystemPrompt();
    const convo = [...messages, { role: "user", content: userText }];
    const toolLogs = [];
    const allowMutationTools = /修改|调参|调整|优化|patch|preview|采纳|批准|写入|apply|执行|变更|改配置|再来一版|继续改/i.test(
      String(userText || "")
    );
    const mutationTools = new Set([
      "agent.propose_patch",
      "agent.preview_patch",
      "agent.apply_patch_with_approval",
      "agent.rollback_run_config"
    ]);
    const prefersRelayGlobal = /^https?:\/\//i.test(apiUrl.trim());
    const defaultContinue = "请继续。若需工具则按约定输出 tool JSON。";
    /** 写入/回滚完成后若仍用「请继续+tool JSON」，模型会再出一轮 patch，形成审批死循环 */
    const finalizeAfterMutation =
      "配置变更已通过工具落盘。请仅用一两句自然语言确认完成（不要输出 JSON 代码块、不要调用任何工具）。在用户未明确要求继续改配置前，禁止主动提出修改建议、禁止追问是否继续调整、禁止再次调用 propose_patch / preview_patch / apply_patch / rollback 相关工具。";
    let continuationSuffix = defaultContinue;

    for (let round = 0; round < maxRounds; round += 1) {
      const transcriptText = convo
        .map((m) => {
          if (m.role === "tool") {
            return `[tool:${m.name}]\n${m.content}`;
          }
          return `[${m.role}] ${m.content}`;
        })
        .join("\n\n");
      const prompt = `${transcriptText}\n\n${continuationSuffix}`;
      const prefersRelay = prefersRelayGlobal;

      if (prefersRelay && round === 0) {
        setMessages((prev) => {
          const idx = prev.length;
          agentSlotIndexRef.current = idx;
          return [...prev, { role: "agent", content: "", reasoningApi: "", toolsUsed: [], streaming: true }];
        });
      } else if (prefersRelay && round > 0) {
        setMessages((prev) => {
          const next = [...prev];
          const idx = agentSlotIndexRef.current;
          if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
            next[idx] = { ...next[idx], streaming: true };
          }
          return next;
        });
      }

      const onDelta = prefersRelay
        ? ({ reply, reasoning }) => {
            setMessages((prev) => {
              const next = [...prev];
              const idx = agentSlotIndexRef.current;
              if (idx >= 0 && idx < next.length && next[idx].role === "agent" && next[idx].streaming) {
                next[idx] = {
                  ...next[idx],
                  content: reply || "",
                  reasoningApi: reasoning || "",
                  streaming: true
                };
              }
              return next;
            });
          }
        : undefined;

      const onRelayFallback = prefersRelay
        ? () => {
            setMessages((prev) => {
              const idx = agentSlotIndexRef.current;
              const next = [...prev];
              if (idx >= 0 && idx < next.length && next[idx].streaming) {
                next.splice(idx, 1);
              }
              agentSlotIndexRef.current = -1;
              return next;
            });
          }
        : undefined;

      let payload;
      let target;
      let reply;
      let displayReply;
      let displayReasoning;
      try {
        const result = await requestAgentWithFallback({
          text: prompt,
          mode: "chat",
          systemPrompt,
          onDelta,
          onRelayFallback
        });
        payload = result.payload;
        target = result.target;
        const parsed = extractReplyAndReasoningFromPayload(payload?.reply || payload);
        reply = parsed.reply;
        const built = buildDisplayReplyAndReasoning(reply, parsed.reasoning);
        displayReply = built.displayReply;
        displayReasoning = built.displayReasoning;
        if (typeof payload?.reasoning === "string" && payload.reasoning.trim() && prefersRelay) {
          setMessages((prev) => {
            const next = [...prev];
            const idx = agentSlotIndexRef.current;
            if (idx >= 0 && idx < next.length && next[idx].role === "agent" && next[idx].streaming) {
              next[idx] = { ...next[idx], reasoningApi: payload.reasoning };
            }
            return next;
          });
        }
      } finally {
        if (prefersRelay) {
          setMessages((prev) => {
            const next = [...prev];
            const idx = agentSlotIndexRef.current;
            if (idx >= 0 && idx < next.length && next[idx].role === "agent" && next[idx].streaming) {
              next[idx] = { ...next[idx], streaming: false };
            }
            return next;
          });
        }
      }

      const toolNames = toolLogs.map((t) => t.call.tool);
      if (prefersRelay && target?.kind === "backend-relay") {
        setMessages((prev) => {
          const next = [...prev];
          const idx = agentSlotIndexRef.current;
          if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
            const mergedReasoning =
              displayReasoning && String(displayReasoning).trim()
                ? displayReasoning
                : next[idx].reasoningApi || "";
            next[idx] = {
              ...next[idx],
              content: displayReply,
              toolsUsed: toolNames,
              ...(mergedReasoning && String(mergedReasoning).trim() ? { reasoningApi: mergedReasoning } : {})
            };
          }
          return next;
        });
      }

      const toolCall = extractToolCallFromText(reply);
      if (!toolCall) {
        return {
          payload,
          reply,
          displayReply,
          displayReasoning,
          target,
          toolLogs,
          streamedRelay: target?.kind === "backend-relay"
        };
      }
      if (mutationTools.has(toolCall.tool) && !allowMutationTools) {
        return {
          payload,
          reply,
          displayReply:
            "当前消息未明确要求修改配置，已阻止自动变更建议与审批流程。若需要修改，请明确说明“请修改哪些参数/请生成并预览 patch”。",
          displayReasoning,
          target,
          toolLogs,
          streamedRelay: target?.kind === "backend-relay"
        };
      }
      if (toolCall.tool === "agent.apply_patch_with_approval") {
        return {
          payload,
          reply,
          displayReply:
            "已拦截自动写入请求。请先在左侧审批面板核对 patch，再使用“让 agent 执行”按钮完成写入。",
          displayReasoning,
          target,
          toolLogs,
          streamedRelay: target?.kind === "backend-relay"
        };
      }
      const execResult = await apiRequest("/api/agent/tools/execute", {
        method: "POST",
        body: JSON.stringify({ tool: toolCall.tool, args: toolCall.args || {} })
      });
      toolLogs.push({ call: toolCall, result: execResult });
      if (toolCall.tool === "agent.apply_patch_with_approval" && execResult && execResult.status === "ok") {
        continuationSuffix = finalizeAfterMutation;
      } else if (toolCall.tool === "agent.rollback_run_config" && execResult && execResult.status === "ok") {
        continuationSuffix = finalizeAfterMutation;
      } else {
        continuationSuffix = defaultContinue;
      }
      const namesAfter = toolLogs.map((t) => t.call.tool);
      if (prefersRelay && agentSlotIndexRef.current >= 0) {
        setMessages((prev) => {
          const next = [...prev];
          const idx = agentSlotIndexRef.current;
          if (idx >= 0 && idx < next.length && next[idx].role === "agent") {
            next[idx] = { ...next[idx], toolsUsed: namesAfter };
          }
          return next;
        });
      }
      convo.push({ role: "assistant", content: reply });
      convo.push({
        role: "tool",
        name: toolCall.tool,
        content: JSON.stringify(execResult, null, 2)
      });
    }
    throw new Error("工具调用达到上限，请缩小问题范围后重试。");
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
      const { payload, reply, displayReply, displayReasoning, target, toolLogs, streamedRelay } = await runAgentWithTools({
        userText: text
      });
      const data = payload;
      if (!streamedRelay) {
        const names = toolLogs.map((t) => t.call.tool);
        setMessages((prev) => [
          ...prev,
          {
            role: "agent",
            content: displayReply,
            toolsUsed: names,
            ...(displayReasoning ? { reasoningApi: displayReasoning } : {})
          }
        ]);
      }
      const terminalOut = formatAgentTerminalOutput(displayReply, toolLogs);
      setOutputText(terminalOut);
      if (target.kind === "openai") {
        toast("已通过 OpenAI 兼容接口完成请求", "success");
      } else if (target.kind === "backend-relay") {
        toast("已通过本地中转完成请求（已绕过浏览器跨域）", "success");
      }
      /* 已成功 apply/rollback 时：不得再跑 syncApprovalFromToolLogs，否则会命中同一条 toolLogs 里更早的 preview_patch，再次打开审批侧栏（死循环） */
      const configMutationDone = toolLogs.some(
        (t) =>
          (t.call?.tool === "agent.apply_patch_with_approval" || t.call?.tool === "agent.rollback_run_config") &&
          t.result?.status === "ok"
      );
      if (!configMutationDone) {
        if (!syncApprovalFromToolLogs(toolLogs, terminalOut)) {
          if (!(await syncProposePatchViaPreview(toolLogs, terminalOut))) {
            await maybeHandlePatch(data, reply);
          }
        }
      }
    } catch (error) {
      setMessages((prev) => [...prev, { role: "agent", content: `请求失败: ${error.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const sendPresetMessage = async (presetText) => {
    const text = (presetText || "").trim();
    if (!text) return;
    if (!apiUrl.trim()) {
      toast("请先填写 Agent API 地址", "warning");
      return;
    }
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    try {
      const { payload, reply, displayReply, displayReasoning, target, toolLogs, streamedRelay } = await runAgentWithTools({
        userText: text
      });
      const data = payload;
      if (!streamedRelay) {
        const names = toolLogs.map((t) => t.call.tool);
        setMessages((prev) => [
          ...prev,
          {
            role: "agent",
            content: displayReply,
            toolsUsed: names,
            ...(displayReasoning ? { reasoningApi: displayReasoning } : {})
          }
        ]);
      }
      const terminalOut = formatAgentTerminalOutput(displayReply, toolLogs);
      setOutputText(terminalOut);
      if (target.kind === "openai") {
        toast("已通过 OpenAI 兼容接口完成请求", "success");
      } else if (target.kind === "backend-relay") {
        toast("已通过本地中转完成请求（已绕过浏览器跨域）", "success");
      }
      const configMutationDone = toolLogs.some(
        (t) =>
          (t.call?.tool === "agent.apply_patch_with_approval" || t.call?.tool === "agent.rollback_run_config") &&
          t.result?.status === "ok"
      );
      if (!configMutationDone) {
        if (!syncApprovalFromToolLogs(toolLogs, terminalOut)) {
          if (!(await syncProposePatchViaPreview(toolLogs, terminalOut))) {
            await maybeHandlePatch(data, reply);
          }
        }
      }
    } catch (error) {
      setMessages((prev) => [...prev, { role: "agent", content: `请求失败: ${error.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  /** 同一轮 toolLogs 中，若某条之后已出现 apply/rollback（失败会抛错不会入栈），则更早的预览/提议不应再驱动审批 UI */
  const hasConfigMutationAfter = (toolLogs, index) => {
    for (let j = index + 1; j < toolLogs.length; j += 1) {
      const c = toolLogs[j].call?.tool;
      if (c === "agent.apply_patch_with_approval" || c === "agent.rollback_run_config") return true;
    }
    return false;
  };

  /** 从工具链中的 agent.preview_patch 结果同步审批票据（与仅解析回复 Markdown 互补） */
  const syncApprovalFromToolLogs = (toolLogs, terminalOut) => {
    if (!Array.isArray(toolLogs) || !toolLogs.length) return false;
    for (let i = toolLogs.length - 1; i >= 0; i -= 1) {
      const { call, result } = toolLogs[i];
      if (call?.tool !== "agent.preview_patch" || !result) continue;
      if (hasConfigMutationAfter(toolLogs, i)) {
        continue;
      }
      const tok = result.approval_token;
      if (!tok) continue;
      if (result.status && result.status !== "ok") continue;
      applyPreviewResponseToUi(result, terminalOut || "", "tool");
      return true;
    }
    return false;
  };

  /**
   * agent.propose_patch 只返回建议 patch，不签发审批令牌；此处代为调用 /api/agent/patch/preview。
   * 工具响应形态：{ status, tool, result: { goal, patch, need_approval } }
   */
  const syncProposePatchViaPreview = async (toolLogs, terminalOut) => {
    if (!Array.isArray(toolLogs) || !toolLogs.length) return false;
    for (let i = toolLogs.length - 1; i >= 0; i -= 1) {
      const { call, result } = toolLogs[i];
      if (call?.tool !== "agent.propose_patch" || !result) continue;
      if (hasConfigMutationAfter(toolLogs, i)) continue;
      // 与 preview_patch 不同，部分中继响应可能省略 status；仅在明确失败时跳过
      if (result.status && result.status !== "ok") {
        continue;
      }
      const inner = result.result && typeof result.result === "object" ? result.result : result;
      const patch = inner && typeof inner === "object" && !Array.isArray(inner) ? inner.patch : null;
      if (!patch || typeof patch !== "object" || Array.isArray(patch) || !Object.keys(patch).length) {
        continue;
      }
      try {
        const preview = await apiRequest("/api/agent/patch/preview", {
          method: "POST",
          body: JSON.stringify({
            patch,
            run_id: (call.args && call.args.run_id) || "default",
            operator: "agent",
            reason: "agent.propose_patch"
          })
        });
        applyPreviewResponseToUi(preview, terminalOut || "", "propose");
        return true;
      } catch (error) {
        toast(`无法从 propose_patch 生成审批预览: ${error.message}`, "error");
        return false;
      }
    }
    return false;
  };

  const applyPreviewResponseToUi = (preview, terminalOut, source) => {
    setApprovalBody(
      `${preview.patch_yaml || ""}\n\n--- merged_preview ---\n${JSON.stringify(preview.merged_preview || {}, null, 2)}`
    );
    setApprovalToken(preview.approval_token || "");
    setApprovalRunId(String(preview.run_id || "default"));
    setApprovalRequestHash(String(preview.request_hash || ""));
    setApprovalOpen(true);
    const suffix =
      source === "tool"
        ? "已通过工具 agent.preview_patch 签发审批票据；请在左侧栏核对 merged_preview，批准后将写入 configs/distill_config.yaml。"
        : source === "propose"
          ? "已根据 agent.propose_patch 的建议调用预览并签发审批票据；请在左侧栏核对 merged_preview，批准后将写入 configs/distill_config.yaml。"
          : "请在左侧栏查看 YAML 与 merged_preview；批准后将写入 configs/distill_config.yaml。";
    setOutputText(`${terminalOut}\n\n# --- Patch 预览 ---\n${suffix}`);
    toast(
      source === "tool"
        ? "已通过工具触发审批预览，可在左侧栏「批准修改训练配置」面板中采纳。"
        : source === "propose"
          ? "已从 propose_patch 生成审批预览，可在左侧栏「批准修改训练配置」面板中采纳。"
          : "已生成配置 patch 预览，可在左侧栏「批准修改训练配置」面板中采纳或让 Agent 执行。",
      "success"
    );
  };

  const maybeHandlePatch = async (result, replyText) => {
    const replyRaw = typeof replyText === "string" ? replyText : "";
    const transcriptEchoLike =
      /^\s*\[assistant\]/.test(replyRaw) ||
      /^\s*\[user\]/.test(replyRaw) ||
      /^\s*\[tool:[^\]]+\]/m.test(replyRaw) ||
      /\n\[tool:[^\]]+\]/.test(replyRaw);
    if (transcriptEchoLike) {
      return;
    }
    const patch = extractPatchFromResult(result, replyText);
    if (!patch) return;
    try {
      const preview = await apiRequest("/api/agent/patch/preview", {
        method: "POST",
        body: JSON.stringify({ patch })
      });
      applyPreviewResponseToUi(preview, formatAgentTerminalOutput(replyText, []), "markdown");
    } catch (error) {
      setMessages((prev) => [...prev, { role: "agent", content: `Patch 校验失败: ${error.message}` }]);
      toast(`Patch 校验失败: ${error.message}`, "error");
    }
  };

  const testAgentApi = async () => {
    if (!apiUrl.trim()) return toast("请先填写 Agent API 地址", "warning");
    try {
      const { target } = await requestAgentWithFallback({ text: "ping", mode: "test", systemPrompt: "" });
      toast(`Agent API 连接成功（${target.kind}）`, "success");
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

  const generateMetricsReport = async () => {
    try {
      let sourcePath = (metricsCsvPath || "").trim();
      if (!sourcePath) {
        const list = await apiRequest("/api/metrics");
        const available = Array.isArray(list.csv_metrics) ? list.csv_metrics.filter((x) => x.has_results) : [];
        if (!available.length) {
          toast("暂无训练结果（未找到 results.csv），无法生成指标快照", "warning");
          return;
        }
        sourcePath = available[0].path;
      }
      const data = await apiRequest(`/api/metrics?source=${encodeURIComponent(sourcePath)}`);
      if (data?.error) {
        toast(`生成报告失败: ${data.error}`, "error");
        return;
      }
      const stats = data?.overview_stats || {};
      const summary = data?.summary_metrics || {};
      const map50 = stats["ov-map50"] || "--";
      const modelSize = stats["ov-model-size"] || "--";
      const params = stats["ov-params"] || "--";
      const flops = stats["ov-flops"] || "--";
      const trainTime = stats["ov-time"] || "--";
      const report = [
        "训练指标分析报告",
        `- mAP50: ${map50}`,
        `- 模型大小: ${modelSize}`,
        `- 参数量: ${params}`,
        `- FLOPs: ${flops}`,
        `- 训练时长: ${trainTime}`,
        "",
        "关键指标摘要:",
        JSON.stringify(summary, null, 2)
      ].join("\n");
      setOutputText(report);
      toast("分析报告已生成，请在右侧输出查看详情。", "success");
    } catch (error) {
      toast(`生成报告失败: ${error.message}`, "error");
    }
  };

  const sendAgentExecuteApproval = async () => {
    if (!approvalToken) {
      toast("暂无待执行的审批票据", "warning");
      return;
    }
    try {
      const execResult = await apiRequest("/api/agent/tools/execute", {
        method: "POST",
        body: JSON.stringify({
          tool: "agent.apply_patch_with_approval",
          args: {
            approval_token: approvalToken,
            run_id: approvalRunId || "default",
            request_hash: approvalRequestHash || undefined,
            operator: "agent-ui"
          }
        })
      });
      setApprovalOpen(false);
      setApprovalToken("");
      setApprovalRunId("default");
      setApprovalRequestHash("");
      setOutputText(
        [
          "# 已通过工具执行写入",
          "agent.apply_patch_with_approval",
          "",
          JSON.stringify(execResult, null, 2)
        ].join("\n")
      );
      toast("已执行 agent.apply_patch_with_approval", "success");
    } catch (error) {
      toast(`执行失败: ${error.message}`, "error");
    }
  };

  return (
    <div className={`tab-panel ${active ? "active" : ""}`} id="panel-agent" aria-hidden={!active}>
      <div className="agent-layout">
        <div className="agent-sidebar">
          <h3 className="sidebar-title"><span className="material-icons">hub</span>连接与工具</h3>
          <div className="agent-settings-card md3-surface-container">
            <h4>外部 API 配置</h4>
            <div className="form-row stacked-row">
              <TextField label="Agent API 地址" value={apiUrl} onChange={setApiUrl} />
              <TextField label="API Token (可选)" value={apiKey} onChange={setApiKey} />
              <TextField label="模型名 / Endpoint ID" value={apiModel} onChange={setApiModel} />
            </div>
            <div className="launch-actions" style={{ marginTop: 12 }}>
              <button className="md-btn md-btn-tonal" onClick={saveConfig}>
                <span className="material-icons">save</span>保存配置
              </button>
              <button className="md-btn md-btn-outlined" onClick={testAgentApi}>
                <span className="material-icons">bolt</span>测试连接
              </button>
            </div>
            <div className="agent-sidebar-aux-tools">
              <button type="button" className="md-btn md-btn-tonal md-btn-compact" onClick={loadSchema} disabled={loading}>
                <span className="material-icons">schema</span>配置结构
              </button>
              <button type="button" className="md-btn md-btn-outlined md-btn-compact" onClick={parseClipboardPatch} disabled={loading}>
                <span className="material-icons">content_paste</span>剪贴板 Patch
              </button>
              <button type="button" className="md-btn md-btn-tonal md-btn-compact" onClick={generateMetricsReport} disabled={loading}>
                <span className="material-icons">summarize</span>指标快照
              </button>
            </div>
          </div>
          <div
            className="agent-approval-sidebar agent-approval-sidebar-frame md3-surface-container"
            role="region"
            aria-labelledby="agent-approval-dialog-title"
          >
            {approvalOpen ? (
              <>
                <h2 id="agent-approval-dialog-title" className="md-dialog-title md3-dialog-headline">
                  批准修改训练配置？
                </h2>
                <p className="md-dialog-support md3-dialog-supporting">
                  确认后请使用下方按钮让 Agent 调用工具写入 configs/distill_config.yaml 并刷新训练配置表单。
                </p>
                <pre className="md-dialog-pre md3-dialog-body-scroll">{approvalBody}</pre>
                <div className="md-dialog-actions md3-dialog-actions agent-approval-sidebar-actions">
                  <button type="button" className="md-btn md-btn-text" onClick={() => setApprovalOpen(false)}>
                    取消
                  </button>
                  <button
                    type="button"
                    className="md-btn md-btn-filled primary md-btn-compact"
                    onClick={sendAgentExecuteApproval}
                    disabled={loading || !approvalToken}
                  >
                    <span className="material-icons">smart_toy</span>让 agent 执行
                  </button>
                </div>
              </>
            ) : (
              <>
                <h2 id="agent-approval-dialog-title" className="md-dialog-title agent-approval-sidebar-idle-title">
                  <span className="material-icons" aria-hidden>
                    verified_user
                  </span>
                  审批区
                </h2>
                <p className="tools-desc agent-approval-sidebar-idle-desc">
                  外部 Agent 返回的 patch 仅预览；通过预览签发票据后，将在此显示 YAML 与 merged_preview，并可使用「让 agent 执行」写入配置。
                </p>
              </>
            )}
          </div>
          <div className="agent-common-tools md3-surface-container">
            <h4 className="tools-title"><span className="material-icons">build</span>常用工具</h4>
            <p className="tools-desc">点击后将对应问句发送到左侧对话，并自动请求 Agent；右侧输出优先展示可复制的终端命令。</p>
            <div className="tools-actions" style={{ flexWrap: "wrap" }}>
              {AGENT_QUICK_PROMPTS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className="md-btn md-btn-outlined md-btn-compact"
                  disabled={loading}
                  onClick={() => sendPresetMessage(p.text)}
                >
                  <span className="material-icons">chat</span>
                  {p.label}
                </button>
              ))}
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
            <div id="agent-chat-messages" ref={chatMessagesRef} className="chat-messages">
              {messages.map((msg, index) => (
                <div key={`${msg.role}-${index}`} className={`chat-message ${msg.role}`}>
                  <div className={`message-avatar ${msg.role === "agent" ? "agent-avatar" : "user-avatar"}`}>
                    <span className="material-icons">{msg.role === "agent" ? "smart_toy" : "person"}</span>
                  </div>
                  <div className="message-content">
                    <ChatBubbleBody
                      role={msg.role}
                      content={msg.content}
                      reasoningApi={msg.reasoningApi}
                      toolsUsed={msg.toolsUsed}
                      streaming={msg.streaming}
                    />
                  </div>
                </div>
              ))}
              {loading && !messages.some((m) => m.streaming) ? (
                <div className="chat-message agent">
                  <div className="message-avatar agent-avatar"><span className="material-icons">smart_toy</span></div>
                  <div className="message-content"><div>处理中...</div></div>
                </div>
              ) : null}
            </div>
            <div className="chat-input-area">
              <div className="chat-input-wrapper">
                <div className={`md-field ${input ? "has-value" : ""}`}>
                  <textarea
                    ref={chatTextareaRef}
                    rows={1}
                    className="md-input"
                    value={input || ""}
                    onChange={(e) => {
                      setInput(e.target.value);
                      resizeChatInput();
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        send();
                      }
                    }}
                    placeholder=" "
                  >
                  </textarea>
                  <label className="md-field-label">输入消息</label>
                </div>
                <button className="md-btn md-btn-filled primary send-btn" type="button" onClick={send} disabled={loading}>
                  <span className="material-icons">send</span>
                </button>
              </div>
            </div>
          </div>
          <div className="agent-status-panel log-card">
            <div className="status-header log-header">
              <h3><span className="material-icons">terminal</span>Agent 输出</h3>
            </div>
            <div id="agent-output" className="agent-output log-container">
              <pre id="agent-output-content">{outputText}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** 仅含外部 Agent 连接字段（侧栏 API 配置）的 JSON，不应触发 distill_config 审批 */
function isLikelyAgentConnectionPayloadOnly(parsed) {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return false;
  const keys = Object.keys(parsed);
  if (keys.length === 0) return false;
  const connectionKeys = new Set([
    "api_url",
    "apiUrl",
    "base_url",
    "baseUrl",
    "agent_api_url",
    "endpoint",
    "endpoint_id",
    "endpointId",
    "model",
    "api_model",
    "apiModel",
    "api_key",
    "apiKey",
    "token",
    "authorization",
    "provider",
    "region"
  ]);
  const hasDistillSection =
    (typeof parsed.distillation === "object" && parsed.distillation !== null && !Array.isArray(parsed.distillation)) ||
    (typeof parsed.training === "object" && parsed.training !== null && !Array.isArray(parsed.training)) ||
    (typeof parsed.output === "object" && parsed.output !== null && !Array.isArray(parsed.output)) ||
    (typeof parsed.wandb === "object" && parsed.wandb !== null && !Array.isArray(parsed.wandb)) ||
    (typeof parsed.patch === "object" && parsed.patch !== null);
  if (hasDistillSection) return false;
  return keys.every((k) => connectionKeys.has(k));
}

function distillPatchFromParsedObject(parsed) {
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  if (isLikelyAgentConnectionPayloadOnly(parsed)) return null;
  if (parsed.patch && typeof parsed.patch === "object") return parsed.patch;
  if (typeof parsed.distillation === "object" && parsed.distillation !== null) return parsed;
  if (typeof parsed.training === "object" && parsed.training !== null) return parsed;
  if (typeof parsed.output === "object" && parsed.output !== null) return parsed;
  if (typeof parsed.wandb === "object" && parsed.wandb !== null) return parsed;
  return null;
}

/** 从混有 shell 注释的代码块中扫描平衡花括号 JSON（例如 powershell 与 JSON 同块时整段无法 JSON.parse） */
function distillPatchFromLooseText(inner) {
  const str = String(inner || "");
  for (let start = 0; start < str.length; start += 1) {
    if (str[start] !== "{") continue;
    let depth = 0;
    for (let j = start; j < str.length; j += 1) {
      const c = str[j];
      if (c === "{") depth += 1;
      else if (c === "}") {
        depth -= 1;
        if (depth === 0) {
          const slice = str.slice(start, j + 1);
          try {
            const parsed = JSON.parse(slice);
            const patch = distillPatchFromParsedObject(parsed);
            if (patch) return patch;
          } catch {
            /* 尝试下一个起始 { */
          }
          break;
        }
      }
    }
  }
  return null;
}

function extractPatchFromResult(result, text) {
  if (result && typeof result.patch === "object" && result.patch !== null) {
    return distillPatchFromParsedObject(result.patch) ? result.patch : null;
  }
  if (result && typeof result.suggested_patch === "object" && result.suggested_patch !== null) {
    return distillPatchFromParsedObject(result.suggested_patch) ? result.suggested_patch : null;
  }
  const raw = typeof text === "string" ? text.trim() : "";
  if (!raw) return null;
  const tryJsonBlock = (inner) => {
    try {
      const parsed = JSON.parse(String(inner || "").trim());
      return distillPatchFromParsedObject(parsed);
    } catch {
      return null;
    }
  };
  const blocks = [...raw.matchAll(/```[a-zA-Z0-9_-]*\s*([\s\S]*?)```/g)];
  for (const m of blocks) {
    const body = m[1] || "";
    const patch = tryJsonBlock(body) || distillPatchFromLooseText(body);
    if (patch) return patch;
  }
  return tryJsonBlock(raw) || distillPatchFromLooseText(raw);
}

export default App;
