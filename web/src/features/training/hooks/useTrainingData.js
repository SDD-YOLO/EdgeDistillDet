import { useEffect } from "react";
import { fetchTrainLogs, fetchTrainStatus } from "../../../api/trainApi";
import { formatTime } from "../../../utils/time";

export function useTrainingData({
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
}) {
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
  }, [running, refreshResumeCandidates, resumeListProjectRef, lastServerRunningRef, setProgress, setRunning, startTimestampRef]);

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
  }, [running, refreshResumeCandidates, resumeListProjectRef]);

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
  }, [running, logOffsetRef, parseMetricsFromLogLines, setLogs, setProgress]);
}
