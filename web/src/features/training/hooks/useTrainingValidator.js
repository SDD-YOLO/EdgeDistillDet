/**
 * useTrainingValidator.js
 * =======================
 * 集成配置验证到 React 组件中
 * 
 * 使用示例：
 * const { validateAndCorrect, showValidationToast } = useTrainingValidator({ toast });
 * 
 * // 在 setNested 之后调用
 * const newForm = validateAndCorrect(updatedForm);
 * setForm(newForm);
 */

import { useCallback } from "react";
import {
  validateTrainingConfig,
  formatViolationMessage,
  getViolationSeverity
} from "../utils/trainingConfigValidator";

export function useTrainingValidator({ toast }) {
  /**
   * 验证配置、应用修正、显示提示
   * @param {Object} form - 要验证的表单
   * @param {Object} options - 选项
   * @returns {Object} 修正后的表单
   */
  const validateAndCorrect = useCallback(
    (form, options = {}) => {
      const { showToast = true, silentWarnings = false } = options;

      if (!form || typeof form !== "object") {
        return form;
      }

      const { form: correctedForm, violations } = validateTrainingConfig(form);

      // 是否有实际修正发生
      const wasModified = JSON.stringify(form) !== JSON.stringify(correctedForm);

      if (showToast && violations.length > 0) {
        // 警告不显示（除非明确要求）
        const errorsOnly = violations.filter(v => !v.isWarning);
        const warningsOnly = violations.filter(v => v.isWarning);

        if (errorsOnly.length > 0 || (!silentWarnings && warningsOnly.length > 0)) {
          const toastSeverity = getViolationSeverity(violations);
          const message = formatViolationMessage(violations);

          if (message) {
            toast(message, toastSeverity);
          }
        }
      }

      return correctedForm;
    },
    [toast]
  );

  /**
   * 手动检查并显示所有校验消息（不修正）
   * @param {Object} form - 表单
   * @returns {Array} 违规项列表
   */
  const checkAndNotify = useCallback(
    (form) => {
      if (!form || typeof form !== "object") {
        return [];
      }

      const { violations } = validateTrainingConfig(form);

      if (violations.length > 0) {
        const severity = getViolationSeverity(violations);
        const message = formatViolationMessage(violations);
        if (message) {
          toast(message, severity);
        }
      }

      return violations;
    },
    [toast]
  );

  /**
   * 获取所有当前违规项（不显示 toast）
   * @param {Object} form - 表单
   * @returns {Object} { form, violations }
   */
  const getViolations = useCallback((form) => {
    return validateTrainingConfig(form);
  }, []);

  return {
    validateAndCorrect,
    checkAndNotify,
    getViolations
  };
}
