import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useInferenceState } from "./useInferenceState";
import { startDisplay, stopDisplay } from "../../../api/trainApi";

vi.mock("../../../api/trainApi", () => ({
  startDisplay: vi.fn(),
  stopDisplay: vi.fn(),
}));

describe("useInferenceState", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("starts and stops inference", async () => {
    startDisplay.mockResolvedValueOnce(undefined);
    stopDisplay.mockResolvedValueOnce(undefined);
    const toast = vi.fn();
    const { result } = renderHook(() => useInferenceState({ toast }));

    await act(async () => {
      await result.current.startInference({ source: "video.mp4" });
    });

    expect(startDisplay).toHaveBeenCalledWith({ source: "video.mp4" });
    expect(result.current.inferRunning).toBe(true);
    expect(toast).toHaveBeenCalledWith("推理已启动", "success");

    await act(async () => {
      await result.current.stopInference();
    });

    expect(stopDisplay).toHaveBeenCalledTimes(1);
    expect(result.current.inferRunning).toBe(false);
    expect(toast).toHaveBeenCalledWith("推理已停止", "info");
  });
});
