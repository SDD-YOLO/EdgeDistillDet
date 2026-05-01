# TrainingPanel 组件拆解指南

## 目标
将 TrainingPanel 从 ~1000 行的单体组件拆解为多个专职的自定义 Hook 和UI组件。

## 完成的拆解

### 1. 自定义 Hooks (已创建)

#### `useTrainingState`
- **职责**：管理训练全生命周期
- **状态**：running, logs, progress, autoScroll
- **方法**：startTraining, stopTraining, downloadTrainLogs
- **位置**：`web/src/features/training/hooks/useTrainingState.js`

#### `useExportState`
- **职责**：管理模型导出
- **状态**：exportRunning, exportLogs, exportStatus, exportWeightCandidates
- **方法**：startExport, stopExport, pollExportStatusAndLogs, clearExportLogs
- **位置**：`web/src/features/training/hooks/useExportState.js`

#### `useInferenceState`
- **职责**：管理推理/显示
- **状态**：inferRunning
- **方法**：startInference, stopInference
- **位置**：`web/src/features/training/hooks/useInferenceState.js`

#### `useResumeState`
- **职责**：管理断点续训
- **状态**：resumeCandidates, selectedResumeIndex
- **方法**：refreshResumeCandidates
- **位置**：`web/src/features/training/hooks/useResumeState.js`

## 后续优化步骤

### Step 1: 在 TrainingPanel 中使用新的 Hooks
```javascript
import { 
  useTrainingState, 
  useExportState, 
  useInferenceState, 
  useResumeState 
} from "./hooks";

function TrainingPanel({ toast, active, view = "training" }) {
  // 使用新的 hooks 替代直接声明
  const trainingState = useTrainingState({ toast });
  const exportState = useExportState({ toast });
  const inferenceState = useInferenceState({ toast });
  const resumeState = useResumeState({ toast });
  
  // 解构需要的变量
  const { running, logs, progress, startTraining, stopTraining } = trainingState;
  const { exportRunning, exportLogs, startExport, stopExport } = exportState;
  // ... etc
}
```

### Step 2: 拆分 UI 子组件
将不同的 view 提取为独立组件：
- `TrainingViewContainer` - 处理训练主界面
- `ExportViewContainer` - 处理导出界面
- `InferenceViewContainer` - 处理推理界面
- `AdvancedViewContainer` - 处理高级参数界面

### Step 3: 配置管理模块化
创建 `useConfigManager` Hook：
- `buildConfigPayload`
- `mergeConfig`
- `saveConfig`
- `loadConfig`

### Step 4: 路径和选择管理
创建 `usePathSelection` Hook：
- `pickLocalPath`
- `refreshRunNameSuggestion`
- `refreshExportWeightCandidates`
- `checkOutputPath`

## 收益

- **代码复用**：同一逻辑可被多个组件使用
- **可测试性**：独立的 Hook 更容易单元测试
- **可维护性**：单一职责原则，每个 Hook 专注一个功能
- **性能**：细粒度的状态更新，避免不必要的重新渲染
- **文件大小**：TrainingPanel 从 ~1000 行减少到 ~400-500 行

## 迁移路线图

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| P1 | 导入新 Hooks 到 TrainingPanel | 1-2 小时 |
| P2 | 拆分 UI 为 View 容器组件 | 2-3 小时 |
| P3 | 配置管理 Hook 化 | 1-2 小时 |
| P4 | 路径选择 Hook 化 | 1 小时 |
| P5 | 单元测试覆盖 | 2-3 小时 |
| P6 | 性能优化和文档 | 1-2 小时 |

## 注意事项

1. **保持向后兼容**：现有 API 和 props 接口不变
2. **增量迁移**：逐个功能迁移，每步都验证
3. **测试覆盖**：为每个新 Hook 添加单元测试
4. **文档更新**：为新 Hooks 添加 JSDoc 注释
