# 跨字段联动校验系统实现总结

## 📋 实现概览

建立了**三层防线**的配置校验体系，确保训练配置始终逻辑自洽：

```
前端层（自动修正 + 提示）
        ↓
    API 层（后端兜底）
        ↓
  配置层（初始化自洽）
```

---

## ✅ 已完成的文件清单

### 1️⃣ **前端验证器** - `trainingConfigValidator.js`
```
📁 web/src/features/training/utils/trainingConfigValidator.js
```

**功能：**
- ✓ 定义 5 条校验规则（TRAINING_CONFIG_RULES）
- ✓ 自动修正逻辑冲突
- ✓ 生成可读的错误消息
- ✓ 区分错误和警告

**导出函数：**
```javascript
validateTrainingConfig(form)           // 主函数：验证 + 修正
formatViolationMessage(violations)     // 格式化消息用于 toast
getViolationSeverity(violations)       // 获取最严重等级
checkRule(ruleKey, form)               // 检查单个规则
applyRuleFix(ruleKey, form)            // 应用单个规则修正
```

---

### 2️⃣ **React Hook** - `useTrainingValidator.js`
```
📁 web/src/features/training/hooks/useTrainingValidator.js
```

**功能：**
- ✓ 封装验证器供 React 组件使用
- ✓ 自动显示 toast 提示
- ✓ 区分严格模式和宽松模式

**导出 Hook：**
```javascript
useTrainingValidator({ toast })

// 返回：
{
  validateAndCorrect(form, options)    // 验证 + 修正 + toast
  checkAndNotify(form)                 // 仅检查 + 通知
  getViolations(form)                  // 仅获取违规项
}
```

---

### 3️⃣ **后端验证器** - `web/schemas.py`
```
📁 web/schemas.py
```

**更新内容：**
```python
# SaveConfigRequest - 保存配置时验证
class SaveConfigRequest(BaseModel):
    @model_validator(mode='after')
    def validate_training_config_consistency(self) -> SaveConfigRequest:
        # 检查跨字段约束
        # 返回 422 Unprocessable Entity 如果验证失败

# TrainStartRequest - 启动训练时验证
class TrainStartRequest(BaseModel):
    @model_validator(mode='after')
    def validate_training_config_consistency(self) -> TrainStartRequest:
        # 防御性检查（可选）
```

---

### 4️⃣ **配置修正** - `configs/distill_config.yaml`
```
📁 configs/distill_config.yaml
```

**修正内容：**
```yaml
distillation:
  distill_end_epoch: 10           # 150 → 10 (与 epochs 对齐)

training:
  warmup_epochs: 3                # 20 → 3 (≤ epochs)
  close_mosaic: 9                 # 10 → 9 (< epochs)
```

**效果：** 新用户首次加载配置即自洽，无初始错误

---

### 5️⃣ **最小化配置模板** - `distill_config_minimal_template.yaml`
```
📁 configs/distill_config_minimal_template.yaml
```

**内容：**
- ✓ 仅保留核心必需字段
- ✓ 详细的校验规则说明
- ✓ 常用场景配置示例
- ✓ 字段约束清晰标注

---

### 6️⃣ **集成指南** - `TRAINING_VALIDATOR_INTEGRATION_GUIDE.md`
```
📁 web/src/features/training/TRAINING_VALIDATOR_INTEGRATION_GUIDE.md
```

**包含：**
- Step 1: 导入验证器
- Step 2: 初始化 hook
- Step 3: 修改 setNested 集成验证
- Step 4-5: 可选的加载/保存验证增强

---

### 7️⃣ **测试用例** - `trainingConfigValidator.test.js`
```
📁 web/src/features/training/utils/trainingConfigValidator.test.js
```

**测试覆盖：**
- ✓ 7 个测试用例
- ✓ 使用示例代码
- ✓ 浏览器控制台可直接运行

---

## 🎯 校验规则详解

| # | 规则 | 约束 | 修正方式 |
|---|------|------|---------|
| 1 | DISTILL_END_WITHIN_EPOCHS | distill_end_epoch ≤ epochs | 自动降低 distill_end_epoch 到 epochs |
| 2 | DISTILL_START_BEFORE_END | distill_start_epoch < distill_end_epoch | 自动调整窗口确保 start < end |
| 3 | WARMUP_WITHIN_EPOCHS | warmup_epochs ≤ epochs | 自动降低 warmup_epochs 到 epochs |
| 4 | CLOSE_MOSAIC_WITHIN_EPOCHS | close_mosaic ≤ epochs | 自动降低 close_mosaic 到 epochs |
| 5 | MEMORY_USAGE_WARNING | batch × imgsz ≤ 13M (估算) | ⚠️ 警告，不修正 |

---

## 🚀 使用流程

### 前端用户流程

```
1. 用户在 TrainingPanel 修改参数
                ↓
2. setNested 触发 → validateAndCorrect 自动修正
                ↓
3. 有违规项 → toast 显示修正内容
                ↓
4. 表单自动更新为修正后的值
                ↓
5. 用户可选择：接受修正 / 重新调整
```

### 后端校验流程

```
1. 前端 POST /config/save 或 POST /train/start
                ↓
2. web/schemas.py model_validator 触发
                ↓
3. 如果验证失败 → 返回 422 Unprocessable Entity + 错误列表
                ↓
4. 如果验证通过 → 继续正常业务流程
```

---

## 📊 典型场景应用

### 场景 1: 用户设置不合理的蒸馏窗口
```javascript
// 用户输入
form.training.epochs = 10
form.distillation.distill_end_epoch = 150

// validateTrainingConfig 触发：
// 检查规则 1 失败 → 自动修正
// distill_end_epoch 降低到 10
// toast: "蒸馏结束 epoch (150) 超过总 epoch (10)，已自动修正"

// 最终表单：
form.distillation.distill_end_epoch = 10
```

### 场景 2: 预热 epoch 超过总 epoch
```javascript
// 用户输入
form.training.epochs = 10
form.training.warmup_epochs = 25

// validateTrainingConfig 触发：
// 检查规则 3 失败 → 自动修正
// warmup_epochs 降低到 10

// 最终表单：
form.training.warmup_epochs = 10
```

### 场景 3: 显存占用过高
```javascript
// 用户输入
form.training.batch = 64
form.training.imgsz = 640

// validateTrainingConfig 触发：
// 检查规则 5：64 × 640 = 26.2M > 13M → 警告
// toast: "⚠️ 显存占用可能过高，建议降低 batch 或 imgsz"

// 表单不修正，由用户决定
```

---

## 🔧 集成 TrainingPanel 的步骤

### Step 1: 导入验证器
```javascript
import { useTrainingValidator } from "./hooks/useTrainingValidator";
```

### Step 2: 初始化 hook
```javascript
function TrainingPanel({ toast, active, view = "training" }) {
  const [form, setForm] = useState(DEFAULT_FORM);
  const { validateAndCorrect } = useTrainingValidator({ toast });
  // ...
}
```

### Step 3: 在 setNested 中集成
```javascript
const setNested = (scope, key, value) => {
  setForm((prev) => {
    const updated = { ...prev, [scope]: { ...prev[scope], [key]: value } };
    return validateAndCorrect(updated, { showToast: true });
  });
};
```

**完整示例见:** `TRAINING_VALIDATOR_INTEGRATION_GUIDE.md`

---

## ⚙️ 配置说明

### validateAndCorrect 选项

```javascript
validateAndCorrect(form, {
  showToast: true,        // 是否显示 toast 提示（默认 true）
  silentWarnings: false   // 是否隐藏警告级别的消息（默认 false）
})
```

### 三种调用场景

| 场景 | showToast | silentWarnings | 用途 |
|------|-----------|----------------|------|
| **用户编辑** | true | false | 实时反馈所有消息 |
| **加载配置** | true | true | 仅显示错误，避免刷屏 |
| **保存验证** | false | false | 静默验证，返回违规项 |

---

## 📈 后续优化方向

### 立即可做

- [ ] 在 saveConfig 前增加 violations 检查（参考 Step 5）
- [ ] 将 validateAndCorrect 集成到所有表单更新点
- [ ] 在 ResumePanel 中也集成验证

### 可选扩展

- [ ] 升级到 react-hook-form + resolver（更强大的表单管理）
- [ ] 添加精确的显存计算（不只是经验阈值）
- [ ] 与 Agent 配置修正流程联动
- [ ] 记录用户修正历史用于学习

---

## 📝 文件位置速查

```
核心功能
├── web/src/features/training/utils/trainingConfigValidator.js
├── web/src/features/training/hooks/useTrainingValidator.js
├── web/schemas.py

配置文件
├── configs/distill_config.yaml (修正默认值)
├── configs/distill_config_minimal_template.yaml (新增模板)

文档和测试
├── web/src/features/training/TRAINING_VALIDATOR_INTEGRATION_GUIDE.md
├── web/src/features/training/utils/trainingConfigValidator.test.js
└── 本文件 (实现总结)
```

---

## 🎓 学习资源

**浏览器控制台快速测试：**
```javascript
// 1. 导入测试
import { runAllTests } from 'web/src/features/training/utils/trainingConfigValidator.test.js'

// 2. 运行
runAllTests()

// 3. 查看结果
```

**单元测试用例：** `trainingConfigValidator.test.js` 提供 7 个完整测试场景

---

## ✨ 核心设计原则

1. **优先修正而非阻断** - 自动修正明显的逻辑错误
2. **清晰的用户提示** - toast 准确说明发生了什么和为什么
3. **分层防线** - 前端即时反馈，后端兜底，配置初始化自洽
4. **最小化改动** - 验证逻辑独立，易于集成
5. **易于扩展** - 新增规则只需加入 TRAINING_CONFIG_RULES 对象

---

**实现完成时间:** 2026-05-02
**维护者:** GitHub Copilot
