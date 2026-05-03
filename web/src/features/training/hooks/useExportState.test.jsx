import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useExportState } from "./useExportState";
import {
  fetchExportStatus,
  fetchExportLogs,
  startExportModel,
  stopExportModel,
} from "../../../api/trainApi";

vi.mock("../../../api/trainApi", () => ({
  fetchExportStatus: vi.fn(),
  fetchExportLogs: vi.fn(),
  startExportModel: vi.fn(),
  stopExportModel: vi.fn(),
}));

describe("useExportState", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-01T10:20:30Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("polls export status and appends logs", async () => {
    fetchExportStatus.mockResolvedValueOnce({
      running: true,
      pid: 42,
      output_path: "/tmp/export",
    });
    fetchExportLogs.mockResolvedValueOnce({
      logs: ["line1", "line2"],
      offset: 2,
    });
    const toast = vi.fn();
    const { result } = renderHook(() => useExportState({ toast }));

    await act(async () => {
      await result.current.pollExportStatusAndLogs();
    });

    expect(fetchExportStatus).toHaveBeenCalledTimes(1);
    expect(fetchExportLogs).toHaveBeenCalledWith({ offset: 0, limit: 120 });
    expect(result.current.exportRunning).toBe(true);
    expect(result.current.exportStatus).toEqual({
      running: true,
      pid: 42,
      output_path: "/tmp/export",
    });
    expect(result.current.exportLogs).toEqual(["line1", "line2"]);
    expect(result.current.exportLogContainerRef.current).toBeNull();
    expect(result.current.exportLogOffsetRef.current).toBe(2);
  });

  it("emits completion toast when export transitions to stopped", async () => {
    fetchExportStatus
      .mockResolvedValueOnce({
        running: true,
        pid: 7,
        output_path: "/tmp/export",
      })
      .mockResolvedValueOnce({
        running: false,
        pid: null,
        output_path: "/tmp/export",
      });
    fetchExportLogs.mockResolvedValue({ logs: [], offset: 0 });
    const toast = vi.fn();
    const { result } = renderHook(() => useExportState({ toast }));

    await act(async () => {
      await result.current.pollExportStatusAndLogs();
      await result.current.pollExportStatusAndLogs();
    });

    expect(toast).toHaveBeenCalledWith("导出完成", "success");
    expect(result.current.exportRunning).toBe(false);
  });

  it("starts export and clears previous logs", async () => {
    startExportModel.mockResolvedValueOnce(undefined);
    const toast = vi.fn();
    const { result } = renderHook(() => useExportState({ toast }));

    act(() => {
      result.current.setExportLogs(["old"]);
      result.current.exportLogOffsetRef.current = 9;
    });

    await act(async () => {
      await result.current.startExport({ config: "distill_config.yaml" });
    });

    expect(startExportModel).toHaveBeenCalledWith({
      config: "distill_config.yaml",
    });
    expect(result.current.exportRunning).toBe(true);
    expect(result.current.exportLogs).toEqual([]);
    expect(result.current.exportLogOffsetRef.current).toBe(0);
    expect(toast).toHaveBeenCalledWith("导出已启动", "success");
  });
});
