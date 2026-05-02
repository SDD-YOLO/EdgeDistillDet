# 后端验证策略说明

**更新时间**: 2026-05-02  
**版本**: v2 - 修订版（轻量化验证）

---

## 📋 概览

采用**分层验证策略**，平衡用户灵活性与系统可靠性：

| 阶段 | 组件 | 职责 | 严格程度 |
|------|------|------|---------|
| **前端** | TrainingPanel | 实时验证 + 自动修正 | ⭐⭐⭐⭐⭐ 最严格 |
| **API 保存** | SaveConfigRequest | 允许保存任何配置 | ⭐ 最宽松 |
| **API 启动** | TrainStartRequest | 基础字段验证 | ⭐⭐ 宽松 |
| **路由处理** | API 路由 | 加载配置后验证 | ⭐⭐⭐ 中等 |
| **训练执行** | 核心模块 | 最终校验 | ⭐⭐⭐⭐ 严格 |

---

## ✅ 各层验证细节

### 1️⃣ 前端层 - TrainingPanel（最严格）

**策略**: 实时验证 + 自动修正

```javascript
const setNested = (scope, key, value) => {
  setForm((prev) => {
    const updated = { ...prev, [scope]: { ...prev[scope], [key]: value } };
    // 自动验证和修正
    return validateAndCorrect(updated, { showToast: true });
  });
};
```

**规则**:
- ✅ distill_end_epoch ≤ epochs → 自动修正
- ✅ warmup_epochs ≤ epochs → 自动修正
- ✅ close_mosaic ≤ epochs → 自动修正
- ✅ batch×imgsz 过大 → ⚠️ 警告

**目的**: 让用户尽早知道问题，自动修正简单错误

---

### 2️⃣ API 保存 - SaveConfigRequest（最宽松）

**策略**: 允许保存任何配置

```python
class SaveConfigRequest(BaseModel):
    name: str = "distill_config.yaml"
    config: dict = Field(default_factory=dict)
    # 无 @model_validator，允许保存
```

**为什么这样设计**:
- 用户可能想保存部分配置或草稿
- 用户可能在修改过程中保存
- 前端已做好验证，后端无需重复

**验证规则**: 无

---

### 3️⃣ API 启动 - TrainStartRequest（基础验证）

**策略**: 仅验证基础字段格式

```python
class TrainStartRequest(BaseModel):
    config: str = "distill_config.yaml"
    mode: str = "distill"
    checkpoint: str | None = None
    allow_overwrite: bool = False

    @model_validator(mode="after")
    def validate_training_start(self) -> TrainStartRequest:
        if not self.config or not isinstance(self.config, str):
            raise ValueError("config 必须是非空字符串")
        
        if self.mode not in ("distill", "resume"):
            raise ValueError(f"mode 必须是 'distill' 或 'resume'，当前: {self.mode}")
        
        return self
```

**验证项目**:
- ✅ config 不能为空
- ✅ mode 必须是 'distill' 或 'resume'

**不验证的项目**:
- ✗ 不检查配置一致性（在路由层做）

---

### 4️⃣ 路由处理 - /train/start 路由（中等严格）

**策略**: 加载配置文件后进行完整验证

```python
# 在 web/routers/train.py 中（后续可实现）
@router.post("/api/train/start")
def start_training(payload: TrainStartRequest):
    # Step 1: 验证请求格式（已在 TrainStartRequest 中做）
    
    # Step 2: 加载配置文件
    config = load_config(payload.config)
    
    # Step 3: 验证配置一致性
    validate_training_config(config)  # 检查跨字段约束
    
    # Step 4: 启动训练
    return start_training_task(config, payload.mode)
```

**验证项目**:
- ✅ distill_end_epoch ≤ epochs
- ✅ distill_start_epoch < distill_end_epoch
- ✅ warmup_epochs ≤ epochs
- ✅ close_mosaic ≤ epochs
- ✅ batch×imgsz 显存检查

---

### 5️⃣ 训练执行 - 核心模块（最严格）

**策略**: 最终防线，确保训练不会失败

- 训练启动前再次验证所有参数
- 异常处理和回退机制
- 详细的错误日志

---

## 🎯 为什么采用这种分层策略？

### 问题 1: 过度验证
❌ **问题**: 如果在 SaveConfigRequest 中做严格验证
- 用户无法保存部分配置
- 用户无法保存草稿
- 用户体验差

✅ **解决**: SaveConfigRequest 不做强制验证

### 问题 2: 验证缺失
❌ **问题**: 如果前端验证逻辑有 bug
- 坏配置会进入后端
- 可能在训练中失败

✅ **解决**: 后端路由和核心模块再次验证

### 问题 3: 用户灵活性
❌ **问题**: 后端验证太严格
- 无法测试边界情况
- 无法调试
- 不够友好

✅ **解决**: 分层验证，前端最严格，保存宽松，启动时检查

---

## 📊 验证流程图

```
用户修改参数
    ↓
TrainingPanel validateAndCorrect()
├─ 检查 5 条规则
├─ 自动修正
└─ 显示 toast 提示
    ↓
[条件] 用户点"保存"
    ↓
POST /api/config/save
    │
    ├─ SaveConfigRequest 验证 (✅ 允许通过)
    └─ 保存成功
    ↓
[条件] 用户点"启动训练"
    ↓
POST /api/train/start
    │
    ├─ TrainStartRequest 验证 (✅ 基础字段检查)
    ├─ 加载配置文件
    ├─ API 路由层验证 (✅ 完整一致性检查)
    └─ 启动训练任务
```

---

## 🧪 测试用例

### SaveConfigRequest - 允许任何配置

```python
# ✅ 有效配置 - 通过
SaveConfigRequest(config={
    "training": {"epochs": 10, "warmup_epochs": 3},
    "distillation": {"distill_end_epoch": 10}
})

# ✅ 无效配置 - 也通过（不验证）
SaveConfigRequest(config={
    "training": {"epochs": 10, "warmup_epochs": 25},
    "distillation": {"distill_end_epoch": 150}
})

# ✅ 空配置 - 通过
SaveConfigRequest(config={})

# ✅ 部分配置 - 通过
SaveConfigRequest(config={"training": {"epochs": 10}})
```

### TrainStartRequest - 基础字段验证

```python
# ✅ 有效请求 - 通过
TrainStartRequest(config="distill_config.yaml", mode="distill")

# ✅ resume 模式 - 通过
TrainStartRequest(config="distill_config.yaml", mode="resume")

# ❌ 无效 mode - 拒绝
TrainStartRequest(config="distill_config.yaml", mode="invalid")
# ValueError: mode 必须是 'distill' 或 'resume'

# ❌ 空 config - 拒绝
TrainStartRequest(config="", mode="distill")
# ValueError: config 必须是非空字符串
```

---

## 🚀 后续实现（可选）

在 `web/routers/train.py` 中添加完整的配置验证：

```python
def validate_config_before_training(config):
    """
    启动训练前的配置验证
    
    调用链：
    /train/start 路由 → 此函数 → 启动训练任务
    """
    training = config.get("training", {})
    distillation = config.get("distillation", {})
    
    epochs = training.get("epochs", 10)
    errors = []
    
    # 检查 distill_end_epoch ≤ epochs
    if distillation.get("distill_end_epoch", 0) > epochs:
        errors.append(...)
    
    # ... 其他检查 ...
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(errors))
```

---

## 📝 总结

| 特性 | 前端 | 保存 | 启动 | 路由 | 训练 |
|------|------|------|------|------|------|
| 完整性验证 | ✅ | ✗ | ✗ | ✅ | ✅ |
| 一致性验证 | ✅ | ✗ | ✗ | ✅ | ✅ |
| 自动修正 | ✅ | ✗ | ✗ | ✗ | ✗ |
| 用户提示 | ✅ | ✗ | ✅ | ✅ | ✅ |
| 严格程度 | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**设计原则**:
- 前端：提前发现，自动修正，友好提示
- 保存：尽可能宽松，允许用户自由操作
- 启动：基础检查，防止明显错误
- 路由：完整验证，确保配置合理
- 训练：最终防线，保证训练不会因配置失败

---

**这种分层策略确保了系统的可靠性，同时给用户充分的灵活性。** ✨
