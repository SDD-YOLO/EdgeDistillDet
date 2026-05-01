import { useCallback, useState } from "react";
import { startDisplay, stopDisplay } from "../../../api/trainApi";

/**
 * 自定义 Hook：管理推理/显示的状态
 * 职责：启动推理、停止推理、推理启用检查
 */
export function useInferenceState({ toast }) {
  const [inferRunning, setInferRunning] = useState(false);

  const startInference = useCallback(async (payload) => {
    try {
      await startDisplay(payload);
      setInferRunning(true);
      toast("推理已启动", "success");
    } catch (error) {
      setInferRunning(false);
      toast(`推理启动失败: ${error.message}`, "error");
    }
  }, [toast]);

  const stopInference = useCallback(async () => {
    try {
      await stopDisplay();
      setInferRunning(false);
      toast("推理已停止", "info");
    } catch (error) {
      toast(`推理停止失败: ${error.message}`, "error");
    }
  }, [toast]);

  return {
    inferRunning,
    setInferRunning,
    startInference,
    stopInference
  };
}
