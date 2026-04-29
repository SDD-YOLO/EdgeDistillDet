import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  checkOutputPath,
  fetchDistillConfig,
  fetchRecentConfig,
  pickDialogPath,
  saveDistillConfig,
  uploadConfigFile
} from "../../api/configApi";
import {
  downloadTrainLogsBlob,
  fetchResumeCandidates,
  fetchExportWeightCandidates,
  startTrain,
  stopTrain,
  startDisplay,
  stopDisplay,
  startExportModel,
  stopExportModel,
  fetchExportStatus,
  fetchExportLogs,
} from "../../api/trainApi";
import { NumberField } from "../../components/forms/NumberField";
import { SelectField } from "../../components/forms/SelectField";
import { TextField } from "../../components/forms/TextField";
import { PathField } from "../../components/forms/PathField";
import { Button } from "../../components/ui/button";
import { DISTILL_CONFIG_UPDATED_EVENT } from "../../constants/distillConfigSync";
import { DEFAULT_FORM, COMPUTE_PRESETS, inferComputeProviderFromConfig } from "../../constants/trainingDefaults";
import {
  BASIC_DISTILLATION_KEYS,
  BASIC_TRAINING_KEYS,
  DISTILLATION_ADVANCED_SECTIONS,
  DISPLAY_ADVANCED_SECTIONS,
  EXPORT_ADVANCED_SECTIONS,
  TRAINING_ADVANCED_SECTIONS
} from "../../constants/advancedParameterCatalog";
import { TrainingLauncher } from "./components/TrainingLauncher";
import { OutputConfigCard } from "./components/OutputConfigCard";
import { ResumeHistoryCard } from "./components/ResumeHistoryCard";
import { DisplayLauncherPanel } from "./components/DisplayLauncherPanel";
import { ExportLauncherPanel } from "./components/ExportLauncherPanel";
import { TrainingWeightsSection } from "./components/TrainingWeightsSection";
import { DistillationCoreParamsCard } from "./components/DistillationCoreParamsCard";
import { TrainingHyperparamsSection } from "./components/TrainingHyperparamsSection";
import { WandbConfigCard } from "./components/WandbConfigCard";
import { TrainingLogsPanel } from "./components/TrainingLogsPanel";
import { AdvancedCardsColumn } from "./components/AdvancedCardsColumn";
import { useTrainingData } from "./hooks/useTrainingData";
import { detectLogLevel } from "../../utils/logging";

function normalizeWandbForUi(wandb) {
  if (!wandb || typeof wandb !== "object") return wandb;
  const next = { ...wandb };
  if (Array.isArray(next.tags)) {
    next.tags = next.tags.map((t) => String(t).trim()).filter(Boolean).join(", ");
  }
  return next;
}

const LEGACY_DISTILLATION_KEYS = ["temperature", "schedule_type", "feat_layer", "alpha", "distill_type"];
const LEGACY_TRAINING_KEYS = ["warm_epochs", "workers", "batch_size", "learning_rate", "grad_clip"];
const LEGACY_OUTPUT_KEYS = ["model_dir", "log_dir"];

function omitKeys(source, keys) {
  const out = { ...(source || {}) };
  keys.forEach((key) => delete out[key]);
  return out;
}

function normalizeAdvancedValueForUi(value) {
  if (value === undefined || value === null) return "";
  if (typeof value === "boolean" || typeof value === "number") return value;
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function extractAdvancedValues(sectionConfig, basicKeys, sections) {
  const config = sectionConfig && typeof sectionConfig === "object" ? sectionConfig : {};
  const supportedKeys = new Set();
  sections.forEach((section) => {
    (section?.params || []).forEach((param) => supportedKeys.add(param.key));
  });
  const out = {};
  Object.entries(config).forEach(([key, value]) => {
    if (basicKeys.has(key)) return;
    if (!supportedKeys.has(key)) return;
    out[key] = normalizeAdvancedValueForUi(value);
  });
  return out;
}

function parseAdvancedValue(raw) {
  if (raw === null || raw === undefined) return undefined;
  if (typeof raw === "boolean" || typeof raw === "number") return raw;
  const text = String(raw).trim();
  if (!text) return undefined;
  const lower = text.toLowerCase();
  if (lower === "true") return true;
  if (lower === "false") return false;
  if (lower === "null" || lower === "none") return null;
  if (/^-?\d+(\.\d+)?$/.test(text)) return Number(text);
  if ((text.startsWith("{") && text.endsWith("}")) || (text.startsWith("[") && text.endsWith("]")) || (text.startsWith("\"") && text.endsWith("\""))) {
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }
  return text;
}

function applyAdvancedOverrides(target, advanced) {
  const merged = { ...(target || {}) };
  Object.entries(advanced || {}).forEach(([key, rawValue]) => {
    const parsed = parseAdvancedValue(rawValue);
    if (parsed === undefined) {
      delete merged[key];
      return;
    }
    merged[key] = parsed;
  });
  return merged;
}

/** 将服务端/Agent 返回的 distill 配置片段合并进表单 state（与 mergeConfig 行为一致） */
function mergeDistillConfigIntoForm(prev, config) {
  if (!config || typeof config !== "object") return prev;
  const inferredProvider = inferComputeProviderFromConfig(config, prev.training?.compute_provider || "local");
  const incomingDistillation = omitKeys(config?.distillation, LEGACY_DISTILLATION_KEYS);
  const incomingTraining = omitKeys(config?.training, LEGACY_TRAINING_KEYS);
  const incomingOutput = omitKeys(config?.output, LEGACY_OUTPUT_KEYS);
  const incomingAdvancedTraining = {
    ...extractAdvancedValues(incomingTraining, BASIC_TRAINING_KEYS, TRAINING_ADVANCED_SECTIONS),
    ...extractAdvancedValues(incomingTraining, BASIC_TRAINING_KEYS, DISPLAY_ADVANCED_SECTIONS),
    ...extractAdvancedValues(incomingTraining, BASIC_TRAINING_KEYS, EXPORT_ADVANCED_SECTIONS),
    ...extractAdvancedValues(config?.advanced?.training, BASIC_TRAINING_KEYS, TRAINING_ADVANCED_SECTIONS),
    ...extractAdvancedValues(config?.advanced?.training, BASIC_TRAINING_KEYS, DISPLAY_ADVANCED_SECTIONS),
    ...extractAdvancedValues(config?.advanced?.training, BASIC_TRAINING_KEYS, EXPORT_ADVANCED_SECTIONS)
  };
  const incomingExportModel = {
    ...extractAdvancedValues(config?.export_model, BASIC_TRAINING_KEYS, EXPORT_ADVANCED_SECTIONS),
    ...extractAdvancedValues(config?.advanced?.training, BASIC_TRAINING_KEYS, EXPORT_ADVANCED_SECTIONS)
  };
  const incomingAdvancedDistillation = {
    ...extractAdvancedValues(incomingDistillation, BASIC_DISTILLATION_KEYS, DISTILLATION_ADVANCED_SECTIONS),
    ...extractAdvancedValues(config?.advanced?.distillation, BASIC_DISTILLATION_KEYS, DISTILLATION_ADVANCED_SECTIONS)
  };
  // Backward compatibility: older UI may have written training.warm_epochs by mistake.
  if (
    Object.prototype.hasOwnProperty.call(incomingTraining, "warm_epochs") &&
    !Object.prototype.hasOwnProperty.call(incomingTraining, "warmup_epochs")
  ) {
    incomingTraining.warmup_epochs = incomingTraining.warm_epochs;
  }
  delete incomingTraining.warm_epochs;
  return {
    ...prev,
    distillation: { ...prev.distillation, ...incomingDistillation },
    training: {
      ...prev.training,
      ...incomingTraining,
      cloud_api: { ...prev.training?.cloud_api, ...(incomingTraining?.cloud_api || {}) },
      dataset_api: { ...prev.training?.dataset_api, ...(incomingTraining?.dataset_api || {}) },
      compute_provider: inferredProvider
    },
    output: { ...prev.output, ...incomingOutput },
    export_model: { ...(prev.export_model || {}), ...incomingExportModel },
    wandb: { ...prev.wandb, ...normalizeWandbForUi(config?.wandb) },
    advanced: {
      training: { ...(prev.advanced?.training || {}), ...incomingAdvancedTraining },
      distillation: { ...(prev.advanced?.distillation || {}), ...incomingAdvancedDistillation }
    }
  };
}

function TrainingPanel({ toast, active, view = "training" }) {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [mode, setMode] = useState("distill");
  const isAdvancedView = view === "advanced";
  const isDisplayView = view === "display";
  const isExportView = view === "export";
  const isTrainingView = view === "training";
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [resumeCandidates, setResumeCandidates] = useState([]);
  const [selectedResumeIndex, setSelectedResumeIndex] = useState(0);
  const [runHint, setRunHint] = useState("将根据项目目录自动推荐可用运行名称。");
  const [outputCheckInfo, setOutputCheckInfo] = useState({ project: "runs", existingNames: [], suggested: "exp1" });
  const [progress, setProgress] = useState({ current: 0, total: 0, elapsed: "--:--:--", expected: "--:--:--" });
  const [autoScroll, setAutoScroll] = useState(true);
  const [inferRunning, setInferRunning] = useState(false);
  const [exportRunning, setExportRunning] = useState(false);
  const [exportAutoScroll, setExportAutoScroll] = useState(true);
  const [exportLogs, setExportLogs] = useState([]);
  const [exportStatus, setExportStatus] = useState({ running: false, pid: null, output_path: null });
  const [exportWeightCandidates, setExportWeightCandidates] = useState([]);
  const [selectedExportWeightIndex, setSelectedExportWeightIndex] = useState(0);

  const logOffsetRef = useRef(0);
  const exportLogOffsetRef = useRef(0);
  const startTimestampRef = useRef(null);
  const exportLogContainerRef = useRef(null);
  /** 上一轮 /status 的 running，用于检测服务端 true→false（不依赖 React state 与 effect 时序） */
  const lastServerRunningRef = useRef(false);
  const resumeListProjectRef = useRef("runs");
  const prevActiveTabRef = useRef(null);
  const prevExportRunningRef = useRef(false);
  const logContainerRef = useRef(null);
  const overlapAlertShownRef = useRef("");
  const outputNameInputRef = useRef(null);
  const pendingOverlapAlertRef = useRef(false);
  /** 与 `configs/distill_config.yaml` 磁盘 mtime 对齐，用于检测 Agent 等外部写入 */
  const distillFileMtimeNsRef = useRef(0);
  const isRemoteApi = form.training.compute_provider === "remote_api";
  const datasetSource = form.training?.dataset_api?.source || (form.training?.dataset_api?.enabled ? "api" : "path");
  const useDatasetApi = isRemoteApi && datasetSource === "api";
  const isResumeMode = mode === "resume";
  const selectedResumeCandidate = resumeCandidates[selectedResumeIndex];
  const isResumeLocked = isResumeMode && Boolean(selectedResumeCandidate);
  const isResumeConfigLocked = isResumeMode;
  const advancedSectionCards = [
    ...TRAINING_ADVANCED_SECTIONS.map((section) => ({ scope: "training", section })),
    ...DISTILLATION_ADVANCED_SECTIONS.map((section) => ({ scope: "distillation", section }))
  ];
  const displaySectionCards = DISPLAY_ADVANCED_SECTIONS.map((section) => ({ scope: "training", section }));
  const exportSectionCards = EXPORT_ADVANCED_SECTIONS.map((section) => ({ scope: "export_model", section }));
  const advancedCardsLeft = advancedSectionCards.filter((_, index) => index % 2 === 0);
  const advancedCardsRight = advancedSectionCards.filter((_, index) => index % 2 === 1);

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

  const setAdvancedValue = (scope, key, value) => {
    setForm((prev) => ({
      ...prev,
      advanced: {
        ...(prev.advanced || {}),
        [scope]: {
          ...(prev.advanced?.[scope] || {}),
          [key]: value
        }
      }
    }));
  };

  const setExportModelValue = (key, value) => {
    setForm((prev) => ({
      ...prev,
      export_model: {
        ...(prev.export_model || {}),
        [key]: value
      }
    }));
  };

  const buildConfigPayload = (sourceForm) => {
    const cloned = JSON.parse(JSON.stringify(sourceForm || {}));
    const payload = {
      distillation: omitKeys(cloned.distillation, LEGACY_DISTILLATION_KEYS),
      training: omitKeys(cloned.training, LEGACY_TRAINING_KEYS),
      output: omitKeys(cloned.output, LEGACY_OUTPUT_KEYS),
      wandb: cloned.wandb || {}
    };
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
    const exportAdvancedKeys = new Set(EXPORT_ADVANCED_SECTIONS.flatMap((section) => (section.params || []).map((param) => param.key)));
    const advancedTraining = { ...(cloned?.advanced?.training || {}) };
    const explicitExportModel = { ...(cloned?.export_model || {}) };
    Object.entries(advancedTraining).forEach(([key, value]) => {
      const explicitValue = explicitExportModel[key];
      if (explicitValue === undefined || explicitValue === "" || exportAdvancedKeys.has(key)) {
        if (value !== undefined) {
          explicitExportModel[key] = value;
        }
      }
    });
    exportAdvancedKeys.forEach((key) => delete advancedTraining[key]);

    payload.export_model = explicitExportModel;
    payload.training = applyAdvancedOverrides(payload.training, advancedTraining);
    payload.distillation = applyAdvancedOverrides(payload.distillation, cloned?.advanced?.distillation || {});
    payload.advanced = {
      training: advancedTraining,
      distillation: cloned?.advanced?.distillation || {}
    };
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
    setForm((prev) => mergeDistillConfigIntoForm(prev, config));
  };

  useEffect(() => {
    const onDistillConfigUpdated = (ev) => {
      const cfg = ev.detail?.config;
      if (!cfg || typeof cfg !== "object") return;
      flushSync(() => {
        setForm((prev) => mergeDistillConfigIntoForm(prev, cfg));
      });
      if (typeof ev.detail?.file_mtime_ns === "number") {
        distillFileMtimeNsRef.current = ev.detail.file_mtime_ns;
      }
    };
    window.addEventListener(DISTILL_CONFIG_UPDATED_EVENT, onDistillConfigUpdated);
    return () => window.removeEventListener(DISTILL_CONFIG_UPDATED_EVENT, onDistillConfigUpdated);
  }, []);

  const fetchDefaultConfig = async () => {
    try {
      const data = await fetchDistillConfig();
      mergeConfig(data.config || {});
      if (typeof data.file_mtime_ns === "number") {
        distillFileMtimeNsRef.current = data.file_mtime_ns;
      }
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

  const refreshResumeCandidates = useCallback(async (project, autoSelect = true) => {
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
            output: { ...prev.output, project: c.project || prev.output?.project || "runs", name: nm }
          };
        });
      }
    } catch {
      setResumeCandidates([]);
    }
  }, []);

  const refreshExportWeightCandidates = useCallback(async (project) => {
    try {
      const result = await fetchExportWeightCandidates(project);
      const candidates = Array.isArray(result.candidates) ? result.candidates : [];
      setExportWeightCandidates(candidates);
      setSelectedExportWeightIndex((idx) => {
        if (!candidates.length) return 0;
        return Math.min(Math.max(0, idx), candidates.length - 1);
      });
    } catch {
      setExportWeightCandidates([]);
      setSelectedExportWeightIndex(0);
    }
  }, []);

  useTrainingData({
    running,
    setRunning,
    setProgress,
    setLogs,
    refreshResumeCandidates,
    resumeListProjectRef,
    lastServerRunningRef,
    startTimestampRef,
    logOffsetRef,
    parseMetricsFromLogLines
  });

  const pollExportStatusAndLogs = useCallback(async () => {
    let status = null;
    try {
      status = await fetchExportStatus();
      setExportStatus(status || { running: false, pid: null, output_path: null });
      setExportRunning(Boolean(status?.running));
    } catch {
      // ignore transient errors during polling
    }

    try {
      const result = await fetchExportLogs({ offset: exportLogOffsetRef.current, limit: 120 });
      let logs = [];
      let offset = exportLogOffsetRef.current;
      if (Array.isArray(result)) {
        logs = result;
      } else if (result && Array.isArray(result.logs)) {
        logs = result.logs;
        offset = typeof result.offset === "number" ? result.offset : offset;
      }
      if (logs.length > 0) {
        exportLogOffsetRef.current = offset + logs.length;
        setExportLogs((prev) => {
          const next = [...prev, ...logs];
          return next.slice(-800);
        });
      }
    } catch {
      // ignore transient polling failures
    }
  }, []);

  useEffect(() => {
    let intervalId = null;
    let isMounted = true;

    if (exportRunning) {
      exportLogOffsetRef.current = exportLogOffsetRef.current || 0;
      pollExportStatusAndLogs();
      intervalId = window.setInterval(() => {
        if (!isMounted) return;
        pollExportStatusAndLogs();
      }, 1500);
    }

    return () => {
      isMounted = false;
      if (intervalId !== null) {
        window.clearInterval(intervalId);
      }
    };
  }, [exportRunning, pollExportStatusAndLogs]);

  useEffect(() => {
    const prev = prevExportRunningRef.current;
    prevExportRunningRef.current = exportRunning;
    if (prev && !exportRunning) {
      const timerId = window.setTimeout(() => {
        pollExportStatusAndLogs();
      }, 1000);
      return () => window.clearTimeout(timerId);
    }
    return undefined;
  }, [exportRunning, pollExportStatusAndLogs]);

  useEffect(() => {
    resumeListProjectRef.current = form.output.project || "runs";
  }, [form.output.project]);

  useEffect(() => {
    const prev = prevActiveTabRef.current;
    prevActiveTabRef.current = active;
    if (active && prev === false) {
      refreshResumeCandidates(resumeListProjectRef.current || "runs", false);
    }
  }, [active]);

  useEffect(() => {
    fetchDefaultConfig().then(() => {
      refreshRunNameSuggestion(DEFAULT_FORM.output.project, DEFAULT_FORM.output.name, true);
      refreshResumeCandidates(DEFAULT_FORM.output.project, false);
      refreshExportWeightCandidates(DEFAULT_FORM.output.project);
    });
  }, []);

  useEffect(() => {
    refreshExportWeightCandidates(form.output.project || "runs");
  }, [form.output.project, refreshExportWeightCandidates]);

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
      const nextProject = selected.project || prev.output?.project || "runs";
      const nextName = String(selected.name || "").trim() || "exp";
      if (prev.output?.project === nextProject && prev.output?.name === nextName) return prev;
      return {
        ...prev,
        output: { ...prev.output, project: nextProject, name: nextName }
      };
    });
  }, [isResumeMode, resumeCandidates, selectedResumeIndex]);

  const scrollLogsToBottom = (el) => {
    if (!el) return;
    window.requestAnimationFrame(() => {
        el.scrollTop = el.scrollHeight;
    });
  };

  useEffect(() => {
    const el = logContainerRef.current;
    if (!el) return;
    if (!autoScroll) return;
    scrollLogsToBottom(el);
  }, [logs, autoScroll]);

  useEffect(() => {
    const el = exportLogContainerRef.current;
    if (!el) return;
    if (!exportAutoScroll) return;
    scrollLogsToBottom(el);
  }, [exportLogs, exportAutoScroll]);

  const currentOutputProject = outputCheckInfo.project || form.output.project || "runs";
  const currentOutputName = (form.output.name || "").trim();

  const startInference = async () => {
    if (inferRunning) return;
    try {
      await startDisplay({
        config: form.config || "distill_config.yaml",
        source: form.advanced?.training?.source,
        weight: form.distillation?.student_weight,
        device: form.training?.device,
        imgsz: form.training?.imgsz,
        conf: form.training?.conf,
        iou: form.training?.iou,
        visualize: form.advanced?.training?.visualize,
        save_txt: form.advanced?.training?.save_txt,
        save_conf: form.advanced?.training?.save_conf,
        save_crop: form.advanced?.training?.save_crop,
        show: form.advanced?.training?.show,
        show_labels: form.advanced?.training?.show_labels,
        show_conf: form.advanced?.training?.show_conf,
        show_boxes: form.advanced?.training?.show_boxes,
        line_width: form.advanced?.training?.line_width,
        output_dir: form.advanced?.training?.output_dir,
      });
      setInferRunning(true);
      toast("推理已开始", "success");
    } catch (error) {
      setInferRunning(false);
      toast(error?.message || "启动可视化推理失败", "error");
    }
  };

  const stopInference = async () => {
    if (!inferRunning) return;
    try {
      await stopDisplay();
      setInferRunning(false);
      toast("推理已停止", "info");
    } catch (error) {
      toast(error?.message || "停止可视化推理失败", "error");
    }
  };

  const normalizeExportString = (value, fallback) => {
    const text = value === undefined || value === null ? String(fallback || "") : String(value);
    const trimmed = text.trim();
    return trimmed === "" ? String(fallback || "").trim() : trimmed;
  };

  const exportPath = normalizeExportString(form.export_model?.export_path, form.advanced?.training?.export_path);
  const exportFormat = normalizeExportString(form.export_model?.format, form.advanced?.training?.format).toLowerCase();
  const exportWeight = String(form.distillation?.student_weight || "").trim();
  const supportedExportFormats = new Set(["onnx", "torchscript"]);
  const isExportReady = Boolean(exportPath && exportWeight) && supportedExportFormats.has(exportFormat);

  const startExport = async () => {
    if (exportRunning) return;
    if (!exportPath) {
      toast("请先填写导出路径", "warning");
      return;
    }
    if (!supportedExportFormats.has(exportFormat)) {
      toast("请选择有效的导出格式", "warning");
      return;
    }

    try {
      exportLogOffsetRef.current = 0;
      setExportLogs([]);
      const res = await startExportModel({
        config: form.config || "distill_config.yaml",
        weight: form.distillation?.student_weight,
        export_path: exportPath,
        format: exportFormat,
        keras: form.export_model?.keras ?? form.advanced?.training?.keras,
        optimize: form.export_model?.optimize ?? form.advanced?.training?.optimize,
        int8: form.export_model?.int8 ?? form.advanced?.training?.int8,
        dynamic: form.export_model?.dynamic ?? form.advanced?.training?.dynamic,
        simplify: form.export_model?.simplify ?? form.advanced?.training?.simplify,
        opset: form.export_model?.opset ?? form.advanced?.training?.opset,
        workspace: form.export_model?.workspace ?? form.advanced?.training?.workspace,
        nms: form.export_model?.nms ?? form.advanced?.training?.nms,
      });
      setExportRunning(true);
      pollExportStatusAndLogs();
      setExportLogs((prev) => [...prev, `导出任务已启动，PID=${res.pid || "unknown"}`]);
      toast("模型导出已开始", "success");
    } catch (error) {
      toast(error?.message || "启动模型导出失败", "error");
    }
  };

  const stopExport = async () => {
    if (!exportRunning) return;
    try {
      await stopExportModel();
      setExportRunning(false);
      setExportLogs((prev) => [...prev, "模型导出已停止。"]);
      toast("模型导出已停止", "info");
    } catch (error) {
      toast(error?.message || "停止模型导出失败", "error");
    }
  };

  const clearExportLogs = () => {
    setExportLogs([]);
  };

  const downloadExportLogs = () => {
    const blob = new Blob([exportLogs.join("\n")], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `export_logs_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.txt`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    toast("导出日志已下载", "success");
  };

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
    const res = await saveDistillConfig(buildConfigPayload(form));
    if (res && typeof res.file_mtime_ns === "number") {
      distillFileMtimeNsRef.current = res.file_mtime_ns;
    }
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
    const defaultProject = DEFAULT_FORM.output.project || "runs";
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
    refreshResumeCandidates(form.output?.project || DEFAULT_FORM.output?.project || "runs", false);
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
      const data = await fetchDistillConfig();
      const serverMtime = Number(data.file_mtime_ns) || 0;
      const localMtime = distillFileMtimeNsRef.current;
      let merged = form;
      const diskDrift =
        (localMtime !== 0 && serverMtime !== localMtime) || (localMtime === 0 && serverMtime > 0);
      if (diskDrift) {
        merged = mergeDistillConfigIntoForm(form, data.config || {});
        flushSync(() => setForm(merged));
      }
      const saveRes = await saveDistillConfig(buildConfigPayload(merged));
      if (saveRes && typeof saveRes.file_mtime_ns === "number") {
        distillFileMtimeNsRef.current = saveRes.file_mtime_ns;
      } else if (serverMtime > 0) {
        distillFileMtimeNsRef.current = serverMtime;
      }
      const body = { config: "distill_config.yaml", mode };
      if (mode === "resume" && resumeCandidates[selectedResumeIndex]?.checkpoint) {
        body.checkpoint = resumeCandidates[selectedResumeIndex].checkpoint;
      }
      await startTrain(body);
      startTimestampRef.current = Date.now();
      logOffsetRef.current = 0;
      setLogs([]);
      setProgress({ current: 0, total: 0, elapsed: "00:00:00", expected: "--:--:--" });
      lastServerRunningRef.current = true;
      setRunning(true);
      toast("训练任务已启动", "success");
    } catch (error) {
      const requiresConfirmation = Boolean(error?.status === 409 && error?.payload?.requires_confirmation);
      if (requiresConfirmation) {
        const project = error.payload?.project || form.output?.project || "runs";
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
        setProgress({ current: 0, total: 0, elapsed: "00:00:00", expected: "--:--:--" });
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
      const blob = await downloadTrainLogsBlob();
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
  const renderAdvancedField = (scope, param) => {
    const legacyValue = form.advanced?.training?.[param.key];
    const exportModelValue = form.export_model?.[param.key];
    const value =
      scope === "export_model"
        ? exportModelValue !== undefined
          ? exportModelValue
          : legacyValue ?? ""
        : form.advanced?.[scope]?.[param.key] ?? "";
    const disabled = running || isResumeConfigLocked;
    const onChangeValue = (next) => {
      if (scope === "export_model") {
        setExportModelValue(param.key, next);
      } else {
        setAdvancedValue(scope, param.key, next);
      }
    };
    if (param.type === "enum") {
      return (
        <SelectField
          label={param.label}
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(next) => onChangeValue(next)}
          options={param.options || []}
          disabled={disabled}
        />
      );
    }
    if (param.type === "number") {
      return (
        <NumberField
          label={param.label}
          value={value === "" ? null : value}
          step="any"
          onChange={(next) => onChangeValue(next === null ? "" : next)}
          disabled={disabled}
        />
      );
    }
    if (param.type === "path") {
      return (
        <PathField
          label={param.label}
          value={String(value || "")}
          onChange={(next) => onChangeValue(next)}
          onBrowse={async () => {
            const next = await pickLocalPath({ kind: "directory", title: "选择导出路径", initialPath: String(value || "") });
            if (next) onChangeValue(next);
          }}
          disabled={disabled}
        />
      );
    }
    return <TextField label={param.label} value={String(value || "")} onChange={(next) => onChangeValue(next)} disabled={disabled} />;
  };

  return (
    <div className={`tab-panel console-module-panel ${active ? "active" : ""}`} id="panel-training" aria-hidden={!active}>
      {isTrainingView ? (
        <section className="train-launcher">
          <TrainingLauncher
            mode={mode}
            running={running}
            onSwitchToDistillMode={switchToDistillMode}
            onSwitchToResumeMode={switchToResumeMode}
            onStartTraining={startTraining}
            onStopTraining={stopTraining}
            onLoadConfigFromFile={loadConfigFromFile}
            onSaveConfig={() => saveConfig().then(() => toast("配置已保存", "success")).catch((e) => toast(e.message, "error"))}
            onResetForm={resetForm}
            isResumeStartDisabled={isResumeStartDisabled}
          />

          <div className="launcher-side">
            <OutputConfigCard
              form={form}
              isResumeMode={isResumeMode}
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
              onSelectCandidate={(idx) => {
                const c = resumeCandidates[idx];
                if (c) {
                  setForm((prev) => ({
                    ...prev,
                    output: { ...prev.output, project: c.project, name: c.name }
                  }));
                }
              }}
            />
          </div>
        </section>
      ) : null}

      <div className="panel-grid">
        {isTrainingView ? (
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
        ) : isDisplayView ? (
          <DisplayLauncherPanel
            inferRunning={inferRunning}
            startInference={startInference}
            stopInference={stopInference}
            displaySectionCards={displaySectionCards}
            renderAdvancedField={renderAdvancedField}
          />
        ) : isExportView ? (
          <ExportLauncherPanel
            exportRunning={exportRunning}
            exportReady={isExportReady}
            startExport={startExport}
            stopExport={stopExport}
            exportAutoScroll={exportAutoScroll}
            setExportAutoScroll={setExportAutoScroll}
            clearExportLogs={clearExportLogs}
            downloadExportLogs={downloadExportLogs}
            exportLogs={exportLogs}
            exportLogContainerRef={exportLogContainerRef}
            exportSectionCards={exportSectionCards}
            renderAdvancedField={renderAdvancedField}
            exportWeight={exportWeight}
            onExportWeightChange={(next) => setNested('distillation', 'student_weight', next)}
            onExportWeightBrowse={async () => {
              try {
                const selected = await pickLocalPath({
                  kind: 'file',
                  title: '选择导出权重文件',
                  initialPath: form.distillation?.student_weight || '',
                  filters: [
                    { name: 'PyTorch Weights', patterns: ['*.pt', '*.pth'] },
                    { name: 'All Files', patterns: ['*.*'] }
                  ]
                });
                if (selected) setNested('distillation', 'student_weight', selected);
              } catch (error) {
                toast(error.message, 'error');
              }
            }}
            exportWeightCandidates={exportWeightCandidates}
            selectedExportWeightIndex={selectedExportWeightIndex}
            onSelectExportWeightCandidate={(nextValue) => {
              const idx = Number(nextValue) || 0;
              setSelectedExportWeightIndex(idx);
              const c = exportWeightCandidates[idx];
              if (c && c.checkpoint) {
                setNested('distillation', 'student_weight', c.checkpoint);
              }
            }}
          />
        ) : (
          <>
            <AdvancedCardsColumn cards={advancedCardsLeft} renderAdvancedField={renderAdvancedField} />
            <AdvancedCardsColumn cards={advancedCardsRight} renderAdvancedField={renderAdvancedField} />
          </>
        )}
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
