import { renderHook, act } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useResumeState } from "./useResumeState";
import { fetchResumeCandidates } from "../../../api/trainApi";

vi.mock("../../../api/trainApi", () => ({
  fetchResumeCandidates: vi.fn()
}));

describe("useResumeState", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads resume candidates and updates the selected candidate", async () => {
    fetchResumeCandidates.mockResolvedValueOnce({
      candidates: [
        { display_name: "exp1", project: "runs", name: "exp1", checkpoint: "runs/exp1/weights/best.pt" }
      ]
    });
    const toast = vi.fn();
    const { result } = renderHook(() => useResumeState({ toast }));

    await act(async () => {
      await result.current.refreshResumeCandidates("runs", true);
    });

    expect(fetchResumeCandidates).toHaveBeenCalledWith("runs");
    expect(result.current.resumeCandidates).toHaveLength(1);
    expect(result.current.selectedResumeIndex).toBe(0);
    expect(result.current.selectedResumeCandidate).toEqual({
      display_name: "exp1",
      project: "runs",
      name: "exp1",
      checkpoint: "runs/exp1/weights/best.pt"
    });
    expect(result.current.resumeListProjectRef.current).toBe("runs");
  });

  it("reports errors while refreshing resume candidates", async () => {
    fetchResumeCandidates.mockRejectedValueOnce(new Error("network down"));
    const toast = vi.fn();
    const { result } = renderHook(() => useResumeState({ toast }));

    await act(async () => {
      await result.current.refreshResumeCandidates("runs", true);
    });

    expect(toast).toHaveBeenCalledWith("刷新续训列表失败: network down", "error");
    expect(result.current.resumeCandidates).toEqual([]);
  });
});