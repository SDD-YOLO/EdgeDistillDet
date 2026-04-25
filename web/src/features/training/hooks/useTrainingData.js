import { useEffect, useRef } from "react";
import { fetchTrainLogs, fetchTrainStatus } from "../../../api/trainApi";
import { formatTime } from "../../../utils/time";

function parseStartTimeToMs(raw) {
  if (raw == null || raw === "") return null;
  const asNumber = Number(raw);
  if (Number.isFinite(asNumber)) {
    // 兼容秒/毫秒/微秒级时间戳
    if (asNumber > 1e14) return Math.floor(asNumber / 1000);
    if (asNumber > 1e11) return Math.floor(asNumber);
    if (asNumber > 0) return Math.floor(asNumber * 1000);
    return null;
  }
  const parsed = Date.parse(String(raw));
  return Number.isFinite(parsed) ? parsed : null;
}

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
  const emaSecPerEpochRef = useRef(null);
  const emaEpochRef = useRef(0);
  const expectedSecRef = useRef(0);

  useEffect(() => {
    if (!running) return undefined;
    const id = window.setInterval(() => {
      const started = startTimestampRef.current;
      if (!started) return;
      const elapsedSec = Math.max(0, Math.floor((Date.now() - started) / 1000));
      setProgress((prev) => {
        return {
          ...prev,
          elapsed: formatTime(elapsedSec),
          // 本地 1s 定时器仅更新耗时，预计总耗时交给状态轮询（避免同一 epoch 内抖动）
          expected: prev.expected
        };
      });
    }, 1000);
    return () => window.clearInterval(id);
  }, [running, setProgress, startTimestampRef]);

  useEffect(() => {
    let statusTimer = null;

    const pollTrainStatus = async () => {
      try {
        const data = await fetchTrainStatus();
        const nextRunning = Boolean(data.running);
        const wasRunning = Boolean(lastServerRunningRef.current);
        if (wasRunning && !nextRunning) {
          refreshResumeCandidates(resumeListProjectRef.current || "runs", false);
        }
        lastServerRunningRef.current = nextRunning;
        setRunning(nextRunning);
        if (nextRunning) {
          if (!startTimestampRef.current) {
            const parsedStart = parseStartTimeToMs(data.start_time);
            if (parsedStart) startTimestampRef.current = parsedStart;
          }
          const statusCurrent = Number(data.current_epoch) || 0;
          const statusTotal = Number(data.total_epochs) || 0;
          const now = Date.now();
          const started = startTimestampRef.current || now;
          const backendElapsedSec = Number(data.elapsed_sec);
          const elapsedSec = Number.isFinite(backendElapsedSec) && backendElapsedSec >= 0
            ? Math.floor(backendElapsedSec)
            : Math.max(0, Math.floor((now - started) / 1000));
          setProgress((prev) => {
            // 轮询状态偶发回退时，保留前端已观测到的更大 epoch，避免预计耗时“卡回 --:--:--”
            const currentEpoch = !wasRunning ? statusCurrent : Math.max(Number(prev.current) || 0, statusCurrent);
            const totalEpoch = !wasRunning ? statusTotal : Math.max(Number(prev.total) || 0, statusTotal);
            let expectedSec = expectedSecRef.current > 0 ? expectedSecRef.current : 0;
            if (currentEpoch > 0 && totalEpoch > 0) {
              const instantSecPerEpoch = elapsedSec / currentEpoch;
              const epochChanged = emaEpochRef.current !== currentEpoch;
              if (emaSecPerEpochRef.current == null || epochChanged) {
                emaSecPerEpochRef.current =
                  emaSecPerEpochRef.current == null
                    ? instantSecPerEpoch
                    : emaSecPerEpochRef.current * 0.7 + instantSecPerEpoch * 0.3;
                emaEpochRef.current = currentEpoch;
                expectedSec = Math.round((emaSecPerEpochRef.current || instantSecPerEpoch) * totalEpoch);
                expectedSecRef.current = expectedSec;
              }
            } else {
              expectedSec = 0;
              expectedSecRef.current = 0;
            }
            return {
              ...prev,
              current: currentEpoch,
              total: totalEpoch,
              elapsed: formatTime(elapsedSec),
              expected: expectedSec > 0 ? formatTime(expectedSec) : "--:--:--"
            };
          });
        } else if (!nextRunning) {
          startTimestampRef.current = null;
          emaSecPerEpochRef.current = null;
          emaEpochRef.current = 0;
          expectedSecRef.current = 0;
          setProgress({ current: 0, total: 0, elapsed: "--:--:--", expected: "--:--:--" });
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
        refreshResumeCandidates(resumeListProjectRef.current || "runs", false);
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
    const project = () => resumeListProjectRef.current || "runs";
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
