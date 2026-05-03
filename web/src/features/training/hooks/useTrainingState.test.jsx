import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useTrainingState } from "./useTrainingState";
import {
  startTrain,
  stopTrain,
  downloadTrainLogsBlob,
} from "../../../api/trainApi";

vi.mock("../../../api/trainApi", () => ({
  startTrain: vi.fn(),
  stopTrain: vi.fn(),
  downloadTrainLogsBlob: vi.fn(),
}));

describe("useTrainingState", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-01T10:20:30Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("starts training and resets local state", async () => {
    startTrain.mockResolvedValueOnce(undefined);
    const toast = vi.fn();
    const { result } = renderHook(() => useTrainingState({ toast }));

    act(() => {
      result.current.setLogs(["old log"]);
      result.current.setProgress({
        current: 2,
        total: 8,
        elapsed: "00:00:01",
        expected: "00:00:10",
      });
      result.current.logOffsetRef.current = 7;
    });

    await act(async () => {
      await result.current.startTraining({ config: "distill_config.yaml" });
    });

    expect(startTrain).toHaveBeenCalledWith({ config: "distill_config.yaml" });
    expect(result.current.running).toBe(true);
    expect(result.current.logs).toEqual([]);
    expect(result.current.progress).toEqual({
      current: 0,
      total: 0,
      elapsed: "--:--:--",
      expected: "--:--:--",
    });
    expect(result.current.logOffsetRef.current).toBe(0);
    expect(result.current.lastServerRunningRef.current).toBe(true);
    expect(result.current.startTimestampRef.current).toBe(Date.now());
    expect(toast).toHaveBeenCalledWith("训练已启动", "success");
  });

  it("stops training and clears the start timestamp", async () => {
    stopTrain.mockResolvedValueOnce(undefined);
    const toast = vi.fn();
    const { result } = renderHook(() => useTrainingState({ toast }));

    act(() => {
      result.current.setRunning(true);
      result.current.startTimestampRef.current = Date.now();
    });

    await act(async () => {
      await result.current.stopTraining();
    });

    expect(stopTrain).toHaveBeenCalledTimes(1);
    expect(result.current.running).toBe(false);
    expect(result.current.startTimestampRef.current).toBeNull();
    expect(toast).toHaveBeenCalledWith("训练已停止", "info");
  });

  it("downloads training logs with a dated filename", async () => {
    downloadTrainLogsBlob.mockResolvedValueOnce(
      new Blob(["hello"], { type: "text/plain" }),
    );
    const toast = vi.fn();
    const click = vi.fn();
    const anchor = { href: "", download: "", click };
    Object.defineProperty(URL, "createObjectURL", {
      value: vi.fn(() => "blob:mock"),
      configurable: true,
    });

    const { result } = renderHook(() => useTrainingState({ toast }));

    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tagName) => {
      if (tagName === "a") return anchor;
      return originalCreateElement(tagName);
    });

    await act(async () => {
      await result.current.downloadTrainLogs();
    });

    expect(downloadTrainLogsBlob).toHaveBeenCalledTimes(1);
    expect(document.createElement).toHaveBeenCalledWith("a");
    expect(anchor.download).toBe("training_logs_2026-05-01.txt");
    expect(click).toHaveBeenCalledTimes(1);
    expect(toast).toHaveBeenCalledWith("日志已下载", "success");
  });
});
