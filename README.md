# EdgeDistillDet

面向边缘场景的微小目标蒸馏训练与评估项目，包含：

- 训练与蒸馏（`Ultralytics + AdaptiveKDTrainer`）
- 统一 CLI（训练/评估/数据分析/边缘剖析）
- 本地 Web 工作台（FastAPI + React）
- 配置驱动的实验流程（`configs/*.yaml`）

## 1. 项目结构

```text
EdgeDistillDet/
├─ main.py                         # CLI 统一入口
├─ core/
│  ├─ distillation/                # 蒸馏训练核心
│  ├─ evaluation/                  # benchmark 评估
│  └─ detection/                   # 推理/检测封装
├─ scripts/                        # train/eval/analyze 脚本入口
├─ utils/                          # 工具层（数据分析/可视化/边缘剖析）
├─ web/
│  ├─ app.py                       # FastAPI 主入口（路由装配）
│  ├─ schemas.py                   # Web 请求模型
│  ├─ services/                    # 路由业务逻辑服务层
│  ├─ src/                         # React 源码
│  ├─ templates/                   # HTML 模板
│  └─ static/                      # 静态资源与前端构建产物
├─ configs/                        # 训练/评估/数据集/Agent 配置
└─ docs/regression_baseline.md     # 重构前行为快照
```

## 2. 环境要求

- Python 3.10+（建议 Conda）
- Node.js 18+（仅 Web 前端开发/构建需要）
- Windows/Linux/macOS 均可
- GPU 训练建议 CUDA 环境可用

## 3. 快速开始

### 3.1 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3.2 构建前端（生产静态资源）

```bash
cd web
npm ci
npm run build
```

构建输出目录：`web/static/dist/`（`app.js` 与 `app.css`）

### 3.3 启动 Web

```bash
python web/app.py
```

默认访问：`http://127.0.0.1:5000`

## 4. CLI 使用

### 4.1 训练

```bash
python main.py train --config configs/distill_config.yaml
```

断点续训：

```bash
python main.py train --config configs/distill_config.yaml --resume auto
```

### 4.2 评估

```bash
python main.py eval --config configs/eval_config.yaml
```

### 4.3 数据集分析与可视化

```bash
python main.py analyze --dataset <your_dataset_path> --output outputs/figures
```

### 4.4 边缘部署剖析

```bash
python main.py profile --weight <model.pt> --device rk3588
```

可选设备：`rk3588`、`ascend310`、`cpu`、`gpu`

## 5. 关键配置说明

### 5.1 蒸馏训练配置 `configs/distill_config.yaml`

- `distillation.*`：教师/学生权重与蒸馏超参数
- `training.*`：数据集、训练轮数、学习率、设备、混合增强
- `output.*`：实验输出目录（`project/name`）
- `wandb.*`：可选 W&B 配置

### 5.2 评估配置 `configs/eval_config.yaml`

- `evaluation.weight_paths`：待评估权重列表
- `evaluation.test_yaml`：数据集 YAML
- `evaluation.gpu_batch/cpu_batch`：不同设备 batch
- `output.csv_path`：评估结果 CSV 保存位置

## 6. Web API 说明（核心）

- `GET /api/configs`：列出可用配置
- `GET /api/config/{name}`：读取配置
- `POST /api/config/save`：保存配置
- `POST /api/train/start`：启动训练
- `POST /api/train/stop`：停止训练
- `GET /api/train/status`：查询训练状态
- `GET /api/metrics`：读取训练指标
- `POST /api/agent/*`：Agent 辅助接口

## 7. 重构后模块化约定

为提升可维护性，本次重构引入以下约定：

- Web 请求模型统一放在 `web/schemas.py`
- Web 业务逻辑优先放到 `web/services/*`
- 共享模型统计逻辑统一放在 `core/model_metrics.py`
- 数据集共享逻辑统一放在 `utils/dataset_common.py`
- 前端通用能力按职责拆分：
  - `web/src/api/client.js`
  - `web/src/hooks/useToast.js`
  - `web/src/constants/trainingDefaults.js`
  - `web/src/utils/logging.js`
  - `web/src/utils/time.js`

## 8. 常见问题

### 8.1 页面空白或静态资源 404

- 确认已执行 `npm run build`
- 确认存在 `web/static/dist/app.js` 和 `web/static/dist/app.css`

### 8.2 训练无法启动

- 检查是否已有残留训练进程
- 检查 `.training.lock` 是否被占用
- 检查 GPU 显存是否不足

### 8.3 评估找不到数据集

- 检查 `eval_config.yaml` 中 `test_yaml` 路径
- 相对路径默认相对于配置文件所在目录与仓库根目录进行解析

### 8.4 日志显示异常或刷新慢

- Web 启动训练时会启用 `EDGE_WEB_LOG=1` 改善管道日志可读性
- 在 Windows 下建议使用项目默认启动方式，避免额外重定向

## 9. 回归验证建议

重构后至少执行以下检查：

1. `python main.py --help` 正常显示四个子命令
2. Web 启动后访问：
   - `GET /api/configs`
   - `GET /api/train/status`
   - `GET /api/metrics`
3. 对比 `docs/regression_baseline.md` 的字段结构与语义是否一致

## 10. 备注

本次结构重构遵循“**不改功能语义**”原则：路径、输入输出字段、训练主流程保持兼容，重点提升模块边界、复用度与可读性。
