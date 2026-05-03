/** Agent 审批写入 distill_config.yaml 后广播，用于同步训练表单内存状态 */
export const DISTILL_CONFIG_UPDATED_EVENT =
  "edgedistill-distill-config-updated";

/** @param {Record<string, unknown>} config @param {string} source @param {Record<string, unknown>} [extra] */
export function broadcastDistillConfigUpdate(config, source, extra = {}) {
  if (!config || typeof config !== "object") return;
  window.dispatchEvent(
    new CustomEvent(DISTILL_CONFIG_UPDATED_EVENT, {
      detail: { config, source, ...extra },
    }),
  );
}
