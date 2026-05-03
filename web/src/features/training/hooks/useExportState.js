import { useCallback, useRef, useState } from "react";
import {
  fetchExportStatus,
  fetchExportLogs,
  startExportModel,
  stopExportModel,
} from "../../../api/trainApi";

/**
 * 自定义 Hook：管理模型导出的全生命周期状态
 * 职责：导出启动、停止、日志轮询、进度跟踪
 */
export function useExportState({ toast }) {
  const [exportRunning, setExportRunning] = useState(false);
  const [exportLogs, setExportLogs] = useState([]);
  const [exportStatus, setExportStatus] = useState({
    running: false,
    pid: null,
    output_path: null,
  });
  const [exportAutoScroll, setExportAutoScroll] = useState(true);
  const [exportWeightCandidates, setExportWeightCandidates] = useState([]);
  const [selectedExportWeightIndex, setSelectedExportWeightIndex] = useState(0);

  const exportLogOffsetRef = useRef(0);
  const exportLogContainerRef = useRef(null);
  const prevExportRunningRef = useRef(false);

  const scrollExportLogsToBottom = (el) => {
    if (el && exportAutoScroll) {
      el.scrollTop = el.scrollHeight;
    }
  };

  const pollExportStatusAndLogs = useCallback(async () => {
    try {
      const [statusRes, logsRes] = await Promise.all([
        fetchExportStatus(),
        fetchExportLogs({ offset: exportLogOffsetRef.current, limit: 120 }),
      ]);

      if (statusRes?.running) {
        setExportRunning(true);
        setExportStatus({
          running: true,
          pid: statusRes?.pid,
          output_path: statusRes?.output_path,
        });
      } else {
        setExportRunning(false);
        setExportStatus({
          running: false,
          pid: null,
          output_path: statusRes?.output_path,
        });
      }

      if (logsRes?.logs && logsRes.logs.length > 0) {
        exportLogOffsetRef.current =
          logsRes.offset || exportLogOffsetRef.current;
        setExportLogs((prev) => [...prev, ...logsRes.logs]);
      }

      const nextRunning = statusRes?.running || false;
      if (prevExportRunningRef.current && !nextRunning) {
        toast("导出完成", "success");
      }
      prevExportRunningRef.current = nextRunning;
    } catch (error) {
      toast(`导出状态查询失败: ${error.message}`, "error");
    }
  }, [toast, exportAutoScroll]);

  const startExport = async (payload) => {
    try {
      await startExportModel(payload);
      setExportLogs([]);
      exportLogOffsetRef.current = 0;
      setExportRunning(true);
      toast("导出已启动", "success");
    } catch (error) {
      toast(`导出启动失败: ${error.message}`, "error");
    }
  };

  const stopExport = async () => {
    try {
      await stopExportModel();
      toast("导出已停止", "info");
    } catch (error) {
      toast(`导出停止失败: ${error.message}`, "error");
    }
  };

  const clearExportLogs = () => {
    setExportLogs([]);
    exportLogOffsetRef.current = 0;
  };

  const downloadExportLogs = () => {
    const blob = new Blob([exportLogs.join("\n")], {
      type: "text/plain;charset=utf-8",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `export_logs_${new Date().toISOString().slice(0, 10)}.txt`;
    link.click();
  };

  return {
    exportRunning,
    setExportRunning,
    exportLogs,
    setExportLogs,
    exportStatus,
    setExportStatus,
    exportAutoScroll,
    setExportAutoScroll,
    exportWeightCandidates,
    setExportWeightCandidates,
    selectedExportWeightIndex,
    setSelectedExportWeightIndex,
    exportLogOffsetRef,
    exportLogContainerRef,
    scrollExportLogsToBottom,
    pollExportStatusAndLogs,
    startExport,
    stopExport,
    clearExportLogs,
    downloadExportLogs,
  };
}
