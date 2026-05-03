EdgeDistillDet

知识蒸馏训练、模型评估与边缘设备性能剖析的集成工具，包含 Web 可视化界面。

---

简介

EdgeDistillDet 是 [SDD-YOLO](https://github.com/SDD-YOLO) 项目框架的第一个可用模块，主要面向小目标检测场景下的知识蒸馏与边缘部署工作流。

项目将蒸馏训练、Benchmark 评估、边缘设备Profiling 和 Web 可视化整合为统一的工具链，通过 YAML 配置驱动，降低实验管理与结果查看的成本。

> 说明：本项目目前处于积极迭代阶段。代码和功能仍在不断完善中，欢迎通过 Issue 反馈问题或建议。

---

环境准备

- Python 3.10+（推荐 3.12）
- Node.js 20（仅前端开发/构建需要）
- CUDA（GPU 训练时需要）
- Windows / Linux / macOS

---

快速开始

安装

```bash
# 1. 克隆仓库
git clone https://github.com/SDD-YOLO/EdgeDistillDet
cd EdgeDistillDet

# 2. Windows 一键安装（自动检查 Python 版本，匹配 CUDA 通道）
installer.bat

# 3. 手动安装（跨平台）
pip install -r requirements.txt

# 验证安装
edgedistilldet --help
```

启动 Web 工作台

```bash
# 构建前端（首次或前端代码更新后需要）
cd web && npm ci && npm run build && cd ..

# 启动服务
python web/app.py
```

访问 http://127.0.0.1:5000

开发模式（前后端热更新）：

- 终端一：`python web/app.py`（后端）
- 终端二：`cd web && npm run dev`（前端，端口 5173）

CLI 使用

```bash
# 蒸馏训练
edgedistilldet train --config configs/distill_config.yaml

# 断点续训
edgedistilldet train --config configs/distill_config.yaml --resume auto

# 模型评估
edgedistilldet eval --config configs/eval_config.yaml

# 数据集分析
edgedistilldet analyze --dataset <数据集路径> --output outputs/figures

# 边缘设备剖析
edgedistilldet profile --weight <模型.pt> --device rk3588
```

支持设备：`rk3588`、`ascend310`、`cpu`、`gpu`

---

项目结构

```
EdgeDistillDet/
├── main.py                 # CLI 统一入口
├── core/
│   ├── distillation/         # 蒸馏训练核心
│   ├── evaluation/           # 评估与 Benchmark
│   └── detection/            # 推理/检测封装
├── scripts/                  # 训练/评估/分析脚本
├── utils/                    # 数据处理、边缘剖析、可视化
├── web/
│   ├── app.py                # FastAPI 主入口
│   ├── routers/              # API 路由
│   ├── services/             # 业务逻辑
│   ├── src/                  # React 前端源码
│   ├── agent_graph/          # AI Agent 工作流（实验性）
│   ├── agent_rag/            # 文档检索（实验性）
│   └── static/dist/          # 前端构建产物
├── configs/                  # YAML 配置
├── tests/                    # pytest 测试
├── pyproject.toml            # 项目元数据
└── requirements.txt          # 依赖入口
```

关键配置文件

文件 用途
`configs/distill_config.yaml` 教师/学生模型、蒸馏参数、训练超参、输出目录
`configs/eval_config.yaml` 待评估权重列表、数据集、设备 batch 大小
`docs/PARAMETER_MAPPING.md` 前后端参数映射矩阵（含高级参数与续训锁定字段）

---

发布新版本

1. 改版本号：修改 `main.py` 中的 `__version__`（`pyproject.toml` 自动读取）
2. 同步前端版本：修改 `web/package.json` 和 `web/package-lock.json` 中的 `version`
3. 写变更记录：在 `docs/CHANGELOG.md` 添加 `## [x.y.z] - YYYY-MM-DD`
4. 构建前端：`cd web && npm ci && npm run build`，提交 `web/static/dist/`
5. 自检：`python -c "from main import __version__; print(__version__)"` 和 `GET /api/version`

---
