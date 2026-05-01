import { useCallback, useRef, useState } from "react";
import { fetchResumeCandidates } from "../../../api/trainApi";

/**
 * 自定义 Hook：管理断点续训相关状态
 * 职责：管理续训候选项、刷新续训列表、续训模式状态
 */
export function useResumeState({ toast }) {
  const [resumeCandidates, setResumeCandidates] = useState([]);
  const [selectedResumeIndex, setSelectedResumeIndex] = useState(0);
  
  const resumeListProjectRef = useRef("runs");

  const refreshResumeCandidates = useCallback(async (project, autoSelect = true) => {
    if (!project) return;
    try {
      resumeListProjectRef.current = project;
      const data = await fetchResumeCandidates(project);
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      setResumeCandidates(candidates);
      if (autoSelect && candidates.length > 0) {
        setSelectedResumeIndex(0);
      }
    } catch (error) {
      toast(`刷新续训列表失败: ${error.message}`, "error");
    }
  }, [toast]);

  const selectedResumeCandidate = resumeCandidates[selectedResumeIndex];

  return {
    resumeCandidates,
    setResumeCandidates,
    selectedResumeIndex,
    setSelectedResumeIndex,
    selectedResumeCandidate,
    resumeListProjectRef,
    refreshResumeCandidates
  };
}
