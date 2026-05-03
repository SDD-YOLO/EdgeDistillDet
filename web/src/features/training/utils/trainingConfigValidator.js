/**
 * trainingConfigValidator.js
 * ===========================
 * 前端层：训练配置跨字段联动校验和自动修正
 *
 * 职责：
 * 1. 验证配置的逻辑自洽性
 * 2. 自动修正明显的冲突（优先修正而非阻断）
 * 3. 收集违规项供 toast 提示
 */

/**
 * 校验规则定义
 */
export const TRAINING_CONFIG_RULES = {
  // 蒸馏结束 epoch 不能超过总 epoch
  DISTILL_END_WITHIN_EPOCHS: {
    key: "distill_end_within_epochs",
    name: "蒸馏结束 epoch",
    check: (form) =>
      form.distillation?.distill_end_epoch <= form.training?.epochs,
    message: (form) =>
      `蒸馏结束 epoch (${form.distillation?.distill_end_epoch}) 超过总 epoch (${form.training?.epochs})，已自动修正`,
    fix: (form) => ({
      ...form,
      distillation: {
        ...form.distillation,
        distill_end_epoch: Math.min(
          form.distillation?.distill_end_epoch || 150,
          form.training?.epochs || 10,
        ),
      },
    }),
  },

  // 蒸馏开始 epoch 必须小于蒸馏结束 epoch
  DISTILL_START_BEFORE_END: {
    key: "distill_start_before_end",
    name: "蒸馏时间段",
    check: (form) =>
      form.distillation?.distill_start_epoch <
      form.distillation?.distill_end_epoch,
    message: (form) =>
      `蒸馏开始 epoch (${form.distillation?.distill_start_epoch}) 不小于结束 epoch (${form.distillation?.distill_end_epoch})，已自动修正`,
    fix: (form) => {
      const endEpoch = form.distillation?.distill_end_epoch || 10;
      const startEpoch = Math.max(
        0,
        Math.min(form.distillation?.distill_start_epoch || 0, endEpoch - 1),
      );
      return {
        ...form,
        distillation: {
          ...form.distillation,
          distill_start_epoch: startEpoch,
          distill_end_epoch: Math.max(startEpoch + 1, endEpoch),
        },
      };
    },
  },

  // 预热 epoch 不能超过总 epoch
  WARMUP_WITHIN_EPOCHS: {
    key: "warmup_within_epochs",
    name: "预热 epoch",
    check: (form) =>
      (form.training?.warmup_epochs || 0) <= (form.training?.epochs || 10),
    message: (form) =>
      `预热 epoch (${form.training?.warmup_epochs}) 超过总 epoch (${form.training?.epochs})，已自动修正`,
    fix: (form) => ({
      ...form,
      training: {
        ...form.training,
        warmup_epochs: Math.min(
          form.training?.warmup_epochs || 3,
          form.training?.epochs || 10,
        ),
      },
    }),
  },

  // 关闭 mosaic epoch 不能超过总 epoch
  CLOSE_MOSAIC_WITHIN_EPOCHS: {
    key: "close_mosaic_within_epochs",
    name: "关闭 mosaic epoch",
    check: (form) =>
      (form.training?.close_mosaic || 0) <= (form.training?.epochs || 10),
    message: (form) =>
      `关闭 mosaic epoch (${form.training?.close_mosaic}) 超过总 epoch (${form.training?.epochs})，已自动修正`,
    fix: (form) => ({
      ...form,
      training: {
        ...form.training,
        close_mosaic: Math.min(
          form.training?.close_mosaic || 10,
          form.training?.epochs || 10,
        ),
      },
    }),
  },

  // batch × imgsz 显存检查（经验阈值）
  MEMORY_USAGE_WARNING: {
    key: "memory_usage_warning",
    name: "显存占用",
    isWarning: true,
    check: (form) => {
      const batch = form.training?.batch || 16;
      const imgsz = form.training?.imgsz || 640;
      // 经验阈值：640×640 batch=32 开始警告
      // 简化计算：pixel_count * batch / 1_000_000 > 13 时警告
      const pixelCount = imgsz * imgsz;
      const estimatedMemory = (pixelCount * batch) / 1_000_000;
      return estimatedMemory <= 13;
    },
    message: (form) => {
      const batch = form.training?.batch || 16;
      const imgsz = form.training?.imgsz || 640;
      const pixelCount = imgsz * imgsz;
      const estimatedMemory = ((pixelCount * batch) / 1_000_000).toFixed(1);
      return `显存占用可能过高 (估算: ${estimatedMemory}M, batch=${batch}, imgsz=${imgsz})，建议降低 batch 或 imgsz`;
    },
    fix: (form) => form, // 警告不修正
  },
};

/**
 * 主验证函数
 * @param {Object} form - 训练配置表单
 * @returns {Object} { form: 修正后的表单, violations: 违规项列表 }
 */
export function validateTrainingConfig(form) {
  if (!form || typeof form !== "object") {
    return { form, violations: [] };
  }

  const violations = [];
  let correctedForm = form;

  // 按规则顺序验证和修正
  Object.values(TRAINING_CONFIG_RULES).forEach((rule) => {
    const passed = rule.check(correctedForm);

    if (!passed) {
      violations.push({
        key: rule.key,
        name: rule.name,
        message: rule.message(correctedForm),
        isWarning: rule.isWarning || false,
        severity: rule.isWarning ? "warning" : "error",
      });

      // 如果有修正函数，执行修正
      if (rule.fix) {
        correctedForm = rule.fix(correctedForm);
      }
    }
  });

  return { form: correctedForm, violations };
}

/**
 * 生成可读的校验摘要（用于 toast）
 * @param {Array} violations - 违规项列表
 * @returns {String} 格式化的提示文本
 */
export function formatViolationMessage(violations) {
  if (!violations || violations.length === 0) {
    return null;
  }

  const errors = violations.filter((v) => !v.isWarning);
  const warnings = violations.filter((v) => v.isWarning);

  const parts = [];

  if (errors.length > 0) {
    parts.push("校验提醒:");
    errors.forEach((v) => {
      parts.push(`• ${v.message}`);
    });
  }

  if (warnings.length > 0) {
    if (parts.length > 0) parts.push("");
    parts.push("⚠️ 警告:");
    warnings.forEach((v) => {
      parts.push(`• ${v.message}`);
    });
  }

  return parts.join("\n");
}

/**
 * 获取最严重的违规等级（用于 toast 类型）
 * @param {Array} violations - 违规项列表
 * @returns {String} 'error' | 'warning' | 'success'
 */
export function getViolationSeverity(violations) {
  if (!violations || violations.length === 0) return "success";
  if (violations.some((v) => !v.isWarning)) return "error";
  return "warning";
}

/**
 * 工具函数：检查特定规则是否违反
 * @param {String} ruleKey - 规则键
 * @param {Object} form - 表单
 * @returns {Boolean}
 */
export function checkRule(ruleKey, form) {
  const rule = Object.values(TRAINING_CONFIG_RULES).find(
    (r) => r.key === ruleKey,
  );
  return rule ? rule.check(form) : true;
}

/**
 * 工具函数：应用特定规则的修正
 * @param {String} ruleKey - 规则键
 * @param {Object} form - 表单
 * @returns {Object} 修正后的表单
 */
export function applyRuleFix(ruleKey, form) {
  const rule = Object.values(TRAINING_CONFIG_RULES).find(
    (r) => r.key === ruleKey,
  );
  return rule && rule.fix ? rule.fix(form) : form;
}
