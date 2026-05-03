/**
 * trainingConfigValidator.test.js
 * ===============================
 * 验证器使用示例和测试案例
 */

import {
  validateTrainingConfig,
  formatViolationMessage,
  getViolationSeverity,
  checkRule,
  applyRuleFix,
} from "./trainingConfigValidator";

/**
 * 使用示例 1: 基础验证
 */
export function example1_basicValidation() {
  const badConfig = {
    training: { epochs: 10, warmup_epochs: 20, close_mosaic: 15 },
    distillation: { distill_start_epoch: 5, distill_end_epoch: 200 },
  };

  const { form: corrected, violations } = validateTrainingConfig(badConfig);

  console.log("❌ 原配置:", badConfig);
  console.log("✅ 修正后:", corrected);
  console.log("📋 违规项:", violations);

  // 输出示例：
  // violations = [
  //   {
  //     key: "distill_end_within_epochs",
  //     name: "蒸馏结束 epoch",
  //     message: "蒸馏结束 epoch (200) 超过总 epoch (10)，已自动修正",
  //     isWarning: false,
  //     severity: "error"
  //   },
  //   ...
  // ]
}

/**
 * 使用示例 2: Toast 消息格式化
 */
export function example2_toastMessage() {
  const badConfig = {
    training: { epochs: 10, warmup_epochs: 25, batch: 64, imgsz: 640 },
    distillation: { distill_end_epoch: 50 },
  };

  const { violations } = validateTrainingConfig(badConfig);
  const message = formatViolationMessage(violations);
  const severity = getViolationSeverity(violations);

  console.log("📢 Toast 消息:");
  console.log(message);
  console.log("🎨 严重级别:", severity); // "error" | "warning" | "success"

  // 输出示例：
  // 校验提醒:
  // • 蒸馏结束 epoch (50) 超过总 epoch (10)，已自动修正
  // • 预热 epoch (25) 超过总 epoch (10)，已自动修正
  //
  // ⚠️ 警告:
  // • 显存占用可能过高 (估算: 26.2M, batch=64, imgsz=640)，建议降低 batch 或 imgsz
}

/**
 * 使用示例 3: React 组件中的集成
 */
export function example3_reactIntegration() {
  // 在 React 组件中：
  // const { validateAndCorrect } = useTrainingValidator({ toast });
  //
  // const setNested = (scope, key, value) => {
  //   setForm((prev) => {
  //     const updated = { ...prev, [scope]: { ...prev[scope], [key]: value } };
  //     return validateAndCorrect(updated, { showToast: true });
  //   });
  // };
}

/**
 * 使用示例 4: 单个规则检查
 */
export function example4_singleRuleCheck() {
  const config = {
    training: { epochs: 10 },
    distillation: { distill_end_epoch: 15 },
  };

  const isValid = checkRule("distill_end_within_epochs", config);
  console.log("✓ 规则检查:", isValid ? "通过" : "失败");

  // 或者直接修正
  const fixed = applyRuleFix("distill_end_within_epochs", config);
  console.log("✓ 修正后:", fixed);
}

// ════════════════════════════════════════════════════════════
// 测试用例
// ════════════════════════════════════════════════════════════

const testCases = [
  {
    name: "✅ 正常配置",
    config: {
      training: {
        epochs: 100,
        warmup_epochs: 5,
        close_mosaic: 90,
        batch: 32,
        imgsz: 640,
      },
      distillation: {
        distill_start_epoch: 0,
        distill_end_epoch: 100,
      },
    },
    expectedViolations: 0,
  },

  {
    name: "❌ 蒸馏结束 > 总 epoch",
    config: {
      training: { epochs: 10 },
      distillation: { distill_end_epoch: 150 },
    },
    expectedViolations: 1,
    expectedFix: { distillation: { distill_end_epoch: 10 } },
  },

  {
    name: "❌ 预热 > 总 epoch",
    config: {
      training: { epochs: 10, warmup_epochs: 20 },
    },
    expectedViolations: 1,
    expectedFix: { training: { warmup_epochs: 10 } },
  },

  {
    name: "❌ close_mosaic > 总 epoch",
    config: {
      training: { epochs: 10, close_mosaic: 15 },
    },
    expectedViolations: 1,
    expectedFix: { training: { close_mosaic: 10 } },
  },

  {
    name: "❌ 蒸馏时间段逆序",
    config: {
      training: { epochs: 100 },
      distillation: {
        distill_start_epoch: 50,
        distill_end_epoch: 30,
      },
    },
    expectedViolations: 1,
  },

  {
    name: "⚠️ 显存过高警告",
    config: {
      training: {
        epochs: 10,
        batch: 64,
        imgsz: 640,
      },
    },
    expectedViolations: 1,
    isWarning: true,
  },

  {
    name: "❌ 多个错误",
    config: {
      training: {
        epochs: 10,
        warmup_epochs: 25,
        close_mosaic: 15,
        batch: 64,
        imgsz: 640,
      },
      distillation: {
        distill_start_epoch: 10,
        distill_end_epoch: 5,
      },
    },
    expectedViolations: 5, // 3 errors + 1 warning
  },
];

/**
 * 运行所有测试
 */
export function runAllTests() {
  console.log("🧪 开始运行测试...\n");

  testCases.forEach((testCase, idx) => {
    const { config, expectedViolations, expectedFix, isWarning, name } =
      testCase;
    const { form: corrected, violations } = validateTrainingConfig(config);

    console.log(`${idx + 1}. ${name}`);
    console.log(
      `   预期违规数: ${expectedViolations}, 实际: ${violations.length}`,
    );
    console.log(
      `   状态: ${
        violations.length === expectedViolations ? "✅ PASS" : "❌ FAIL"
      }`,
    );

    if (violations.length > 0) {
      violations.forEach((v) => {
        console.log(`   - [${v.severity}] ${v.message}`);
      });
    }

    if (expectedFix) {
      console.log(`   修正验证: ${JSON.stringify(expectedFix, null, 2)}`);
    }
    console.log();
  });
}

/**
 * 导出测试统计
 */
export function getTestStats() {
  return {
    totalCases: testCases.length,
    errorCases: testCases.filter((t) => !t.isWarning).length,
    warningCases: testCases.filter((t) => t.isWarning).length,
  };
}

// 在浏览器控制台中测试：
// import { runAllTests } from './trainingConfigValidator.test.js'
// runAllTests()
