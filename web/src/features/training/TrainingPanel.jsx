import { useEffect, useRef, useState } from "react";
import {
  checkOutputPath,
  fetchDistillConfig,
  fetchRecentConfig,
  pickDialogPath,
  saveDistillConfig,
  uploadConfigFile
} from "../../api/configApi";
import { fetchResumeCandidates, fetchTrainLogs, fetchTrainStatus, startTrain, stopTrain } from "../../api/trainApi";
import { M3Select } from "../../components/forms/M3Select";
import { NumberField } from "../../components/forms/NumberField";
import { PathField } from "../../components/forms/PathField";
import { SelectField } from "../../components/forms/SelectField";
import { TextField } from "../../components/forms/TextField";
import { DEFAULT_FORM, COMPUTE_PRESETS, inferComputeProviderFromConfig } from "../../constants/trainingDefaults";
import { detectLogLevel } from "../../utils/logging";
import { formatTime } from "../../utils/time";

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

  const logOffsetRef = useRef(0);
  const startTimestampRef = useRef(null);
  /** 上一轮 /status 的 running，用于检测服务端 true→false（不依赖 React state 与 effect 时序） */
  const lastServerRunningRef = useRef(false);
  const resumeListProjectRef = useRef("runs/distill");
  const prevActiveTabRef = useRef(null);
  const logContainerRef = useRef(null);
  const overlapAlertShownRef = useRef("");
  const outputNameInputRef = useRef(null);
  const pendingOverlapAlertRef = useRef(false);
  const isRemoteApi = form.training.compute_provider === "remote_api";
  const datasetSource = form.training?.dataset_api?.source || (form.training?.dataset_api?.enabled ? "api" : "path");
  const useDatasetApi = isRemoteApi && datasetSource === "api";
  const isResumeMode = mode === "resume";

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
    const result = await pickDialogPath({
      kind,
      title,
      initial_path: initialPath || "",
      filters
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
      const data = await fetchDistillConfig();
      mergeConfig(data.config || {});
    } catch {
      // 使用默认配置
    }
  };

  const refreshRunNameSuggestion = async (project, currentName, forceDefault = false) => {
    try {
      const info = await checkOutputPath(project);
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
      const result = await fetchResumeCandidates(project);
      const candidates = Array.isArray(result.candidates) ? result.candidates : [];
      setResumeCandidates(candidates);
      setSelectedResumeIndex((idx) => {
        if (!candidates.length) return 0;
        if (autoSelect) return 0;
        return Math.min(Math.max(0, idx), candidates.length - 1);
      });
      if (autoSelect && candidates.length) {
        const c = candidates[0];
        setForm((prev) => {
          const nm = String(c.name || "").trim() || "exp";
          return {
            ...prev,
            output: { ...prev.output, project: c.project || prev.output?.project || "runs/distill", name: nm }
          };
        });
      }
    } catch {
      setResumeCandidates([]);
    }
  };

  useEffect(() => {
    resumeListProjectRef.current = form.output.project || "runs/distill";
  }, [form.output.project]);

  useEffect(() => {
    const prev = prevActiveTabRef.current;
    prevActiveTabRef.current = active;
    if (active && prev === false) {
      refreshResumeCandidates(resumeListProjectRef.current || "runs/distill", false);
    }
  }, [active]);

  useEffect(() => {
    fetchDefaultConfig().then(() => {
      refreshRunNameSuggestion(DEFAULT_FORM.output.project, DEFAULT_FORM.output.name, true);
      refreshResumeCandidates(DEFAULT_FORM.output.project, false);
    });
  }, []);

  useEffect(() => {
    if (!isResumeMode) return;
    if (resumeCandidates.length === 0) {
      setForm((prev) => {
        if (!String(prev.output?.name || "").trim()) return prev;
        return { ...prev, output: { ...prev.output, name: "" } };
      });
      return;
    }
    const selected = resumeCandidates[selectedResumeIndex];
    if (!selected) {
      setForm((prev) => {
        if (!String(prev.output?.name || "").trim()) return prev;
        return { ...prev, output: { ...prev.output, name: "" } };
      });
      return;
    }
    setForm((prev) => {
      const nextProject = selected.project || prev.output?.project || "runs/distill";
      const nextName = String(selected.name || "").trim() || "exp";
      if (prev.output?.project === nextProject && prev.output?.name === nextName) return prev;
      return {
        ...prev,
        output: { ...prev.output, project: nextProject, name: nextName }
      };
    });
  }, [isResumeMode, resumeCandidates, selectedResumeIndex]);

  useEffect(() => {
    let statusTimer = null;

    const pollTrainStatus = async () => {
      try {
        const data = await fetchTrainStatus();
        const nextRunning = Boolean(data.running);
        if (lastServerRunningRef.current && !nextRunning) {
          refreshResumeCandidates(resumeListProjectRef.current || "runs/distill", false);
        }
        lastServerRunningRef.current = nextRunning;
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
    };

    const statusIntervalMs = () => {
      if (running) return 2000;
      return document.hidden ? 20000 : 5000;
    };

    const scheduleStatus = () => {
      if (statusTimer != null) {
        window.clearInterval(statusTimer);
        statusTimer = null;
      }
      statusTimer = window.setInterval(pollTrainStatus, statusIntervalMs());
    };

    const onVisibilityForStatus = () => {
      if (!document.hidden) {
        pollTrainStatus();
        refreshResumeCandidates(resumeListProjectRef.current || "runs/distill", false);
      }
      scheduleStatus();
    };

    document.addEventListener("visibilitychange", onVisibilityForStatus);
    scheduleStatus();
    return () => {
      document.removeEventListener("visibilitychange", onVisibilityForStatus);
      if (statusTimer != null) window.clearInterval(statusTimer);
    };
  }, [running]);

  useEffect(() => {
    if (!running) return undefined;
    const project = () => resumeListProjectRef.current || "runs/distill";
    const tick = () => {
      if (document.hidden) return;
      refreshResumeCandidates(project(), false);
    };
    tick();
    const id = window.setInterval(tick, 12000);
    return () => window.clearInterval(id);
  }, [running]);

  useEffect(() => {
    if (!running) return undefined;
    let logTimer = null;

    const pollLogs = async () => {
      try {
        const offset = Number.isFinite(logOffsetRef.current) ? logOffsetRef.current : 0;
        const data = await fetchTrainLogs({ offset, limit: 120 });
        if (!Array.isArray(data.logs) || data.logs.length === 0) return;
        logOffsetRef.current = data.offset + data.logs.length;
        setLogs((prev) => [...prev, ...data.logs].slice(-800));
        parseMetricsFromLogLines(data.logs, setProgress);
      } catch {
        // 静默处理
      }
    };

    const scheduleLogs = () => {
      if (logTimer != null) {
        window.clearInterval(logTimer);
        logTimer = null;
      }
      if (document.hidden) return;
      logTimer = window.setInterval(pollLogs, 1200);
    };

    const onVisibilityForLogs = () => {
      if (!document.hidden) {
        pollLogs();
      }
      scheduleLogs();
    };

    document.addEventListener("visibilitychange", onVisibilityForLogs);
    scheduleLogs();
    return () => {
      document.removeEventListener("visibilitychange", onVisibilityForLogs);
      if (logTimer != null) window.clearInterval(logTimer);
    };
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
  const isOutputPathOverlap = !isResumeMode && Boolean(currentOutputName && outputCheckInfo.existingNames.includes(currentOutputName));
  const renderedHint = isResumeMode
    ? `断点续练输出目录跟随历史记录: ${currentOutputProject}${currentOutputName ? `/${currentOutputName}` : " — 暂无可用历史运行"}`
    : isOutputPathOverlap
    ? `路径重合: ${currentOutputProject}/${currentOutputName}`
    : `建议输出目录: ${currentOutputProject}/${outputCheckInfo.suggested || "exp1"}`;

  useEffect(() => {
    if (!isOutputPathOverlap) {
      pendingOverlapAlertRef.current = false;
      return;
    }
    if (!isOutputPathOverlap || !currentOutputName) return;
    if (isResumeMode) {
      return;
    }
    const overlapKey = `${currentOutputProject}/${currentOutputName}`;
    if (overlapAlertShownRef.current === overlapKey) return;
    const isFocused = document.activeElement === outputNameInputRef.current;
    if (isFocused) {
      pendingOverlapAlertRef.current = true;
      return;
    }
    pendingOverlapAlertRef.current = false;
  }, [currentOutputProject, currentOutputName, isOutputPathOverlap, isResumeMode]);

  const saveConfig = async () => {
    await saveDistillConfig(buildConfigPayload(form));
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

  const switchToDistillMode = () => {
    if (mode === "distill") return;
    const defaultProject = DEFAULT_FORM.output.project || "runs/distill";
    overlapAlertShownRef.current = "";
    pendingOverlapAlertRef.current = false;
    setForm((prev) => ({
      ...prev,
      output: {
        ...prev.output,
        project: defaultProject,
        name: ""
      }
    }));
    setOutputCheckInfo({ project: defaultProject, existingNames: [], suggested: "exp1" });
    setMode("distill");
    refreshRunNameSuggestion(defaultProject, "", true);
    refreshResumeCandidates(defaultProject, false);
  };

  const switchToResumeMode = () => {
    if (mode === "resume") return;
    setMode("resume");
    refreshResumeCandidates(form.output?.project || DEFAULT_FORM.output?.project || "runs/distill", false);
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
      await startTrain(body);
      startTimestampRef.current = Date.now();
      logOffsetRef.current = 0;
      setLogs([]);
      lastServerRunningRef.current = true;
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
        await startTrain(overwriteBody);
        startTimestampRef.current = Date.now();
        logOffsetRef.current = 0;
        setLogs([]);
        lastServerRunningRef.current = true;
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
      await stopTrain();
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
      const result = await uploadConfigFile({ content, name: file.name });
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
      const result = await fetchRecentConfig();
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
    <div className={`tab-panel console-module-panel ${active ? "active" : ""}`} id="panel-training" aria-hidden={!active}>
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
              <input type="radio" checked={mode === "distill"} onChange={switchToDistillMode} disabled={running} />
              <div className="mode-card">
                <span className="material-icons mode-icon">school</span>
                <div className="mode-text">
                  <strong>蒸馏训练</strong>
                  <span>知识蒸馏训练，训练完成后自动评估模型性能</span>
                </div>
              </div>
            </label>
            <label className={`mode-option ${mode === "resume" ? "selected" : ""}`}>
              <input type="radio" checked={mode === "resume"} onChange={switchToResumeMode} disabled={running} />
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
                    if (isResumeMode) return;
                    setNested("output", "project", project);
                    refreshRunNameSuggestion(project || "runs/distill", form.output.name, true);
                    refreshResumeCandidates(project || "runs/distill", false);
                  }}
                  onBrowse={async () => {
                    if (isResumeMode) return;
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

          <div className="config-card config-card-compact">
            <h3 className="card-header">蒸馏核心参数</h3>
            <div className="form-grid training-form-grid">
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
              {logs.length === 0 ? <div className="log-line info log-empty-placeholder">暂无日志输出</div> : null}
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
        </div>
      </div>
    </div>
  );
}

function parseMetricsFromLogLines(lines, setProgress) {
  const last = Array.isArray(lines) ? lines.slice(-20) : [];
  last.forEach((line) => {
    const text = String(line);
    const ep = text.match(/\[EPOCH_PROGRESS\]\s+epoch=(\d+)\s+total=(\d+)\s+loss=([\d.]+)\s+kd=([\d.]+)\s+alpha=([\d.]+)\s+temp=([\d.]+)/i);
    if (ep) {
      setProgress((prev) => ({ ...prev, current: Number(ep[1]), total: Number(ep[2]) }));
    }
  });
}

export default TrainingPanel;
