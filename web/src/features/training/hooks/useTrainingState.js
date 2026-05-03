import { useCallback, useRef, useState } from "react";
import {
  startTrain,
  stopTrain,
  downloadTrainLogsBlob,
} from "../../../api/trainApi";

/**
 * 自定义 Hook：管理训练的全生命周期状态
 * 职责：训练启动、停止、日志管理、进度追踪
 */
export function useTrainingState({ toast }) {
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [progress, setProgress] = useState({
    current: 0,
    total: 0,
    elapsed: "--:--:--",
    expected: "--:--:--",
  });
  const [autoScroll, setAutoScroll] = useState(true);

  const logOffsetRef = useRef(0);
  const startTimestampRef = useRef(null);
  const lastServerRunningRef = useRef(false);
  const logContainerRef = useRef(null);

  const scrollLogsToBottom = (el) => {
    if (el && autoScroll) {
      el.scrollTop = el.scrollHeight;
    }
  };

  const startTraining = useCallback(
    async (payload) => {
      try {
        await startTrain(payload);
        setRunning(true);
        setLogs([]);
        logOffsetRef.current = 0;
        setProgress({
          current: 0,
          total: 0,
          elapsed: "--:--:--",
          expected: "--:--:--",
        });
        startTimestampRef.current = Date.now();
        lastServerRunningRef.current = true;
        toast("训练已启动", "success");
      } catch (error) {
        setRunning(false);
        toast(`训练启动失败: ${error.message}`, "error");
      }
    },
    [toast],
  );

  const stopTraining = useCallback(async () => {
    try {
      await stopTrain();
      setRunning(false);
      startTimestampRef.current = null;
      toast("训练已停止", "info");
    } catch (error) {
      toast(`训练停止失败: ${error.message}`, "error");
    }
  }, [toast]);

  const downloadTrainLogs = useCallback(async () => {
    try {
      const blob = await downloadTrainLogsBlob();
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `training_logs_${new Date()
        .toISOString()
        .slice(0, 10)}.txt`;
      link.click();
      toast("日志已下载", "success");
    } catch (error) {
      toast(`下载日志失败: ${error.message}`, "error");
    }
  }, [toast]);

  return {
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
    scrollLogsToBottom,
    startTraining,
    stopTraining,
    downloadTrainLogs,
  };
}
