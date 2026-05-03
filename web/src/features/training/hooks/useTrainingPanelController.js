import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  fetchDistillConfig,
  fetchRecentConfig,
  pickDialogPath,
  saveDistillConfig,
} from "../../../api/configApi";
import {
  fetchResumeCandidates,
  startTrain,
  startDisplay,
  stopDisplay,
  startExportModel,
  stopExportModel,
} from "../../../api/trainApi";
import { DISTILL_CONFIG_UPDATED_EVENT } from "../../../constants/distillConfigSync";
import { DEFAULT_FORM } from "../../../constants/trainingDefaults";
import {
  buildConfigPayload,
  mergeDistillConfigIntoForm,
} from "../utils/configManager";
import { useTrainingData } from "./useTrainingData";
import { useTrainingState } from "./useTrainingState";
import { useExportState } from "./useExportState";
import { useInferenceState } from "./useInferenceState";
import { useResumeState } from "./useResumeState";

function parseMetricsFromLogLines(lines, setProgress) {
  const last = Array.isArray(lines) ? lines.slice(-20) : [];
  last.forEach((line) => {
    const text = String(line);
    const ep = text.match(
      /\[EPOCH_PROGRESS\]\s+epoch=(\d+)\s+total=(\d+)\s+loss=([\d.]+)\s+kd=([\d.]+)\s+alpha=([\d.]+)\s+temp=([\d.]+)/i,
    );
    if (ep) {
      setProgress((prev) => ({
        ...prev,
        current: Number(ep[1]),
        total: Number(ep[2]),
      }));
    }
  });
}

export function useTrainingPanelController({ toast }) {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [runHint, setRunHint] =
    useState("将根据项目目录自动推荐可用运行名称。");
  const distillFileMtimeNsRef = useRef(0);

  const trainingState = useTrainingState({ toast });
  const exportState = useExportState({ toast });
  const inferenceState = useInferenceState({ toast });
  const resumeState = useResumeState({ toast });

  const {
    running,
    setRunning,
    logs,
    setLogs,
    progress,
    setProgress,
    autoScroll,
    setAutoScroll,
    logOffsetRef,
    startTimestampRef,
    lastServerRunningRef,
    logContainerRef,
    scrollLogsToBottom: trainingScrollToBottom,
  } = trainingState;

  const {
    exportRunning,
    setExportRunning,
    exportAutoScroll,
    setExportAutoScroll,
    exportLogs,
    setExportLogs,
    exportStatus,
    setExportStatus,
    exportLogOffsetRef,
    exportLogContainerRef,
    pollExportStatusAndLogs,
  } = exportState;

  const { inferRunning, setInferRunning } = inferenceState;

  const { refreshResumeCandidates, resumeListProjectRef } = resumeState;

  const pickLocalPath = useCallback(
    async ({
      kind = "file",
      title = "选择路径",
      initialPath = "",
      filters = [],
    } = {}) => {
      const result = await pickDialogPath({
        kind,
        title,
        initial_path: initialPath || "",
        filters,
      });
      return String(result?.path || "");
    },
    [],
  );

  const getValueByPath = useCallback(
    (path) => {
      if (!path) return "";
      return (
        String(path)
          .split(".")
          .reduce(
            (acc, key) =>
              acc && Object.prototype.hasOwnProperty.call(acc, key)
                ? acc[key]
                : undefined,
            form,
          ) ?? ""
      );
    },
    [form],
  );

  const setValueByPath = useCallback((path, value) => {
    if (!path) return;
    const segments = String(path).split(".");
    setForm((prev) => {
      const next = structuredClone(prev);
      let cursor = next;
      for (let index = 0; index < segments.length - 1; index += 1) {
        const key = segments[index];
        if (
          cursor[key] === undefined ||
          cursor[key] === null ||
          typeof cursor[key] !== "object"
        ) {
          cursor[key] = {};
        }
        cursor = cursor[key];
      }
      cursor[segments[segments.length - 1]] = value;
      return next;
    });
  }, []);

  const previewPayload = useMemo(() => buildConfigPayload(form), [form]);

  const mergeConfig = useCallback((config) => {
    setForm((prev) => mergeDistillConfigIntoForm(prev, config));
  }, []);

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
    window.addEventListener(
      DISTILL_CONFIG_UPDATED_EVENT,
      onDistillConfigUpdated,
    );
    return () =>
      window.removeEventListener(
        DISTILL_CONFIG_UPDATED_EVENT,
        onDistillConfigUpdated,
      );
  }, []);

  const fetchDefaultConfig = useCallback(async () => {
    try {
      const data = await fetchDistillConfig();
      mergeConfig(data.config || {});
      if (typeof data.file_mtime_ns === "number") {
        distillFileMtimeNsRef.current = data.file_mtime_ns;
      }
    } catch {
      // keep defaults
    }
  }, [mergeConfig]);

  const loadConfigFromFile = useCallback(async () => {
    const data = await fetchRecentConfig();
    if (data?.config) {
      mergeConfig(data.config);
    }
  }, [mergeConfig]);

  const resetForm = useCallback(async () => {
    setForm(DEFAULT_FORM);
    await fetchDefaultConfig();
  }, [fetchDefaultConfig]);

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
    parseMetricsFromLogLines,
  });

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
  }, [exportRunning, pollExportStatusAndLogs, exportLogOffsetRef]);

  useEffect(() => {
    resumeListProjectRef.current = form.output.project || "runs";
  }, [form.output.project, resumeListProjectRef]);

  useEffect(() => {
    fetchDefaultConfig().then(() => {
      refreshResumeCandidates(DEFAULT_FORM.output.project, false);
    });
  }, [fetchDefaultConfig, refreshResumeCandidates]);

  useEffect(() => {
    const el = logContainerRef.current;
    if (!el || !autoScroll) return;
    trainingScrollToBottom(el);
  }, [logs, autoScroll, logContainerRef, trainingScrollToBottom]);

  useEffect(() => {
    const el = exportLogContainerRef.current;
    if (!el || !exportAutoScroll) return;
    el.scrollTop = el.scrollHeight;
  }, [exportLogs, exportAutoScroll, exportLogContainerRef]);

  const startInference = useCallback(async () => {
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
  }, [form, inferRunning, setInferRunning, toast]);

  const stopInference = useCallback(async () => {
    if (!inferRunning) return;
    try {
      await stopDisplay();
      setInferRunning(false);
      toast("推理已停止", "info");
    } catch (error) {
      toast(error?.message || "停止可视化推理失败", "error");
    }
  }, [inferRunning, setInferRunning, toast]);

  const normalizeExportString = useCallback((value, fallback) => {
    const text =
      value === undefined || value === null
        ? String(fallback || "")
        : String(value);
    const trimmed = text.trim();
    return trimmed === "" ? String(fallback || "").trim() : trimmed;
  }, []);

  const exportPath = normalizeExportString(
    form.export_model?.export_path,
    form.advanced?.training?.export_path,
  );
  const exportFormat = normalizeExportString(
    form.export_model?.format,
    form.advanced?.training?.format,
  ).toLowerCase();
  const exportWeight = String(form.distillation?.student_weight || "").trim();
  const supportedExportFormats = new Set(["onnx", "torchscript"]);
  const isExportReady =
    Boolean(exportPath && exportWeight) &&
    supportedExportFormats.has(exportFormat);

  const startExport = useCallback(async () => {
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
        optimize:
          form.export_model?.optimize ?? form.advanced?.training?.optimize,
        int8: form.export_model?.int8 ?? form.advanced?.training?.int8,
        dynamic: form.export_model?.dynamic ?? form.advanced?.training?.dynamic,
        simplify:
          form.export_model?.simplify ?? form.advanced?.training?.simplify,
        opset: form.export_model?.opset ?? form.advanced?.training?.opset,
        workspace:
          form.export_model?.workspace ?? form.advanced?.training?.workspace,
        nms: form.export_model?.nms ?? form.advanced?.training?.nms,
      });
      setExportRunning(true);
      pollExportStatusAndLogs();
      setExportLogs((prev) => [
        ...prev,
        `导出任务已启动，PID=${res.pid || "unknown"}`,
      ]);
      toast("模型导出已开始", "success");
    } catch (error) {
      toast(error?.message || "启动模型导出失败", "error");
    }
  }, [
    exportFormat,
    exportLogOffsetRef,
    exportPath,
    exportRunning,
    form,
    pollExportStatusAndLogs,
    setExportLogs,
    setExportRunning,
    toast,
  ]);

  const stopExport = useCallback(async () => {
    if (!exportRunning) return;
    try {
      await stopExportModel();
      setExportRunning(false);
      setExportLogs((prev) => [...prev, "模型导出已停止。"]);
      toast("模型导出已停止", "info");
    } catch (error) {
      toast(error?.message || "停止模型导出失败", "error");
    }
  }, [exportRunning, setExportLogs, setExportRunning, toast]);

  const renderedHint = runHint;
  const progressPercent =
    progress.total > 0
      ? Math.min(100, (progress.current / progress.total) * 100)
      : 0;

  const saveConfig = useCallback(async () => {
    const res = await saveDistillConfig(previewPayload);
    if (res && typeof res.file_mtime_ns === "number") {
      distillFileMtimeNsRef.current = res.file_mtime_ns;
    }
  }, [previewPayload]);

  const startTraining = useCallback(async () => {
    if (running) return;
    try {
      const payload = buildConfigPayload(form);
      const res = await startTrain({ config: payload, mode: "distill" });
      setRunning(true);
      setProgress({ current: 0, total: 0 });
      setLogs((prev) => [...prev, `训练已开始，PID=${res.pid || "unknown"}`]);
      toast("训练已开始", "success");
    } catch (error) {
      setRunning(false);
      toast(error?.message || "启动训练失败", "error");
    }
  }, [form, running, setLogs, setProgress, setRunning, toast]);

  const stopTraining = useCallback(async () => {
    if (!running) return;
    try {
      await stopTrain();
      setRunning(false);
      setLogs((prev) => [...prev, "训练已停止。"]);
      toast("训练已停止", "info");
    } catch (error) {
      toast(error?.message || "停止训练失败", "error");
    }
  }, [running, setLogs, setRunning, toast]);

  useEffect(() => {
    const handler = (ev) => {
      const act = ev.detail?.action;
      switch (act) {
        case "startTraining":
          startTraining();
          break;
        case "stopTraining":
          stopTraining();
          break;
        case "saveConfig":
          saveConfig();
          break;
        case "loadConfig":
          loadConfigFromFile();
          break;
        case "resetForm":
          resetForm();
          break;
        case "startDisplay":
          startInference();
          break;
        case "stopDisplay":
          stopInference();
          break;
        case "startExport":
          startExport();
          break;
        case "stopExport":
          stopExport();
          break;
        case "expandAll":
        case "collapseAll":
          // UI-only controls handled by the config panel; ignore here.
          break;
        default:
          break;
      }
    };
    window.addEventListener("config:action", handler);
    return () => window.removeEventListener("config:action", handler);
  }, [
    startTraining,
    stopTraining,
    saveConfig,
    loadConfigFromFile,
    resetForm,
    startInference,
    stopInference,
    startExport,
    stopExport,
  ]);

  return {
    form,
    setForm,
    runHint,
    renderedHint,
    progressPercent,
    previewPayload,
    getValueByPath,
    setValueByPath,
    pickLocalPath,
    saveConfig,
    loadConfigFromFile,
    resetForm,
    startTraining,
    stopTraining,
    startInference,
    stopInference,
    startExport,
    stopExport,
    running,
    logs,
    progress,
    autoScroll,
    setAutoScroll,
    logOffsetRef,
    startTimestampRef,
    lastServerRunningRef,
    logContainerRef,
    exportRunning,
    exportAutoScroll,
    setExportAutoScroll,
    exportLogs,
    exportLogContainerRef,
    exportStatus,
    setExportStatus,
    inferRunning,
    setInferRunning,
    resumeCandidates: resumeState.resumeCandidates,
    selectedResumeIndex: resumeState.selectedResumeIndex,
    setSelectedResumeIndex: resumeState.setSelectedResumeIndex,
    refreshResumeCandidates,
    resumeListProjectRef,
    exportWeightCandidates: exportState.exportWeightCandidates,
    setExportWeightCandidates: exportState.setExportWeightCandidates,
    selectedExportWeightIndex: exportState.selectedExportWeightIndex,
    setSelectedExportWeightIndex: exportState.setSelectedExportWeightIndex,
    exportLogOffsetRef,
    pollExportStatusAndLogs,
    exportReady: isExportReady,
    trainingScrollToBottom,
    setLogs,
    setProgress,
    setRunning,
    setExportLogs,
    setExportRunning,
  };
}
