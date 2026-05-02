# 跨字段验证系统 - 快速参考卡

## 🚀 5 分钟快速开始

### 什么是跨字段验证？
> 自动检查和修正训练参数之间的逻辑冲突
> 
> 例如：蒸馏结束 epoch (150) 不能大于总训练 epoch (10)

---

## 📂 核心文件

| 文件 | 功能 | 使用人群 |
|------|------|---------|
| `trainingConfigValidator.js` | 验证规则引擎 | 开发者 |
| `useTrainingValidator.js` | React Hook 封装 | 前端开发者 |
| `web/schemas.py` | 后端 Pydantic 验证 | 后端开发者 |
| `distill_config.yaml` | 修正后的默认配置 | 所有人 |
| `INTEGRATION_GUIDE.md` | 集成步骤 | 前端开发者 |

---

## 🎯 5 条校验规则

```
❌ distill_end_epoch > epochs       → 自动降低 distill_end_epoch
❌ distill_start_epoch ≥ distill_end → 自动调整时间窗口  
❌ warmup_epochs > epochs             → 自动降低 warmup_epochs
❌ close_mosaic > epochs              → 自动降低 close_mosaic
⚠️  batch × imgsz > 13M              → 警告提示，不修正
```

---

## 💻 前端集成（3 步）

### Step 1: 导入
```javascript
import { useTrainingValidator } from "./hooks/useTrainingValidator";
```

### Step 2: 初始化
```javascript
const { validateAndCorrect } = useTrainingValidator({ toast });
```

### Step 3: 在 setNested 中使用
```javascript
const setNested = (scope, key, value) => {
  setForm((prev) => {
    const updated = { ...prev, [scope]: { ...prev[scope], [key]: value } };
    return validateAndCorrect(updated, { showToast: true });
  });
};
```

**就这么简单！** 剩下的自动处理 ✨

---

## 🐍 后端验证（已完成）

```python
# web/schemas.py 已自动添加

class SaveConfigRequest(BaseModel):
    @model_validator(mode='after')
    def validate_training_config_consistency(self):
        # 自动检查所有规则，违反返回 422
        return self

class TrainStartRequest(BaseModel):
    @model_validator(mode='after')
    def validate_training_config_consistency(self):
        return self
```

---

## 🧪 测试和调试

### 在浏览器控制台快速测试

```javascript
// 导入验证函数
import { validateTrainingConfig, formatViolationMessage } from 
  'web/src/features/training/utils/trainingConfigValidator'

// 测试一个坏配置
const bad = {
  training: { epochs: 10, warmup_epochs: 25 },
  distillation: { distill_end_epoch: 100 }
}

const { form: fixed, violations } = validateTrainingConfig(bad)

console.log("修正前:", bad)
console.log("修正后:", fixed)
console.log("消息:", formatViolationMessage(violations))
```

### 运行完整测试套件

```javascript
import { runAllTests } from 
  'web/src/features/training/utils/trainingConfigValidator.test.js'
  
runAllTests() // 输出 7 个测试用例的结果
```

---

## 📋 API 参考

### validateTrainingConfig(form)

```javascript
const { form: corrected, violations } = validateTrainingConfig(form)

// violations 示例：
[
  {
    key: "warmup_within_epochs",
    name: "预热 epoch",
    message: "预热 epoch (20) 超过总 epoch (10)，已自动修正",
    isWarning: false,
    severity: "error"
  }
]
```

### useTrainingValidator({ toast })

```javascript
const { 
  validateAndCorrect,     // (form, options) → corrected form
  checkAndNotify,         // (form) → violations + toast
  getViolations           // (form) → violations
} = useTrainingValidator({ toast })
```

---

## 💡 使用场景

### 场景 1: 用户修改参数
```
用户改 epochs → setNested 触发 
  → validateAndCorrect 自动修正
  → toast 提示修正内容
  → 表单更新为修正值
```

### 场景 2: 加载配置文件
```
加载配置 → mergeConfig 验证
  → silentWarnings=true 仅显示错误
  → 避免刷屏
```

### 场景 3: 保存配置
```
点保存 → validateAndCorrect(showToast=false)
  → 如有错误 → 阻止保存 + 显示错误
  → 如无错误 → 保存配置
```

---

## ⚠️ 常见问题

**Q: 为什么要自动修正而不是直接阻断？**
A: 改善用户体验。用户经常不知道参数间有依赖关系，自动修正 + 提示更友好。

**Q: 是否可以禁用自动修正？**
A: 可以。在 validateAndCorrect 中设置 `options.fix = false`。

**Q: 后端验证会拒绝我的请求吗？**
A: 会。如果坏配置绕过前端，后端会返回 422 + 错误列表。

**Q: 我要怎样扩展规则？**
A: 在 `TRAINING_CONFIG_RULES` 对象中添加新规则对象，格式如下：
```javascript
export const TRAINING_CONFIG_RULES = {
  MY_NEW_RULE: {
    key: "my_new_rule",
    name: "规则名称",
    check: (form) => /* 检查逻辑 */,
    message: (form) => /* 错误消息 */,
    fix: (form) => /* 修正逻辑 */
  }
}
```

---

## 📊 配置自洽性对比

### ❌ 修改前（旧 distill_config.yaml）
```yaml
training:
  epochs: 10
  warmup_epochs: 20          # ❌ 超过总 epoch！
  close_mosaic: 10           # ⚠️ 等于 epochs

distillation:
  distill_end_epoch: 150     # ❌ 远超总 epoch！
```

### ✅ 修改后（新 distill_config.yaml）
```yaml
training:
  epochs: 10
  warmup_epochs: 3           # ✓ ≤ epochs
  close_mosaic: 9            # ✓ < epochs

distillation:
  distill_end_epoch: 10      # ✓ ≤ epochs
```

---

## 🎓 详细文档

- **详细实现文档**: `docs/TRAINING_CONFIG_VALIDATION_IMPLEMENTATION.md`
- **集成指南**: `web/src/features/training/TRAINING_VALIDATOR_INTEGRATION_GUIDE.md`
- **测试用例**: `web/src/features/training/utils/trainingConfigValidator.test.js`

---

## 🔗 相关文件清单

```
✅ 已创建/已修改
├── web/src/features/training/utils/trainingConfigValidator.js ✨ NEW
├── web/src/features/training/hooks/useTrainingValidator.js ✨ NEW
├── web/src/features/training/TRAINING_VALIDATOR_INTEGRATION_GUIDE.md ✨ NEW
├── web/src/features/training/utils/trainingConfigValidator.test.js ✨ NEW
├── web/schemas.py (已添加 @model_validator) 📝 MODIFIED
├── configs/distill_config.yaml (修正默认值) 📝 MODIFIED
├── configs/distill_config_minimal_template.yaml ✨ NEW
└── docs/TRAINING_CONFIG_VALIDATION_IMPLEMENTATION.md ✨ NEW

❓ 待集成（需要你手动修改）
└── web/src/features/training/TrainingPanel.jsx
    - 需要在 setNested 中调用 validateAndCorrect
    - 见 TRAINING_VALIDATOR_INTEGRATION_GUIDE.md
```

---

**让我们一起确保用户的配置永远是逻辑自洽的！** ✨
