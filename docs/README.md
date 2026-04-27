# EdgeDistillDet

面向边缘设备的小目标检测蒸馏训练与评估工具。

**核心能力**： teacher-student 蒸馏训练 · 边缘设备性能剖析 · 一键评估 · Web 可视化工作台

---

## 目录

- [环境准备](#环境准备)
- [快速开始](#快速开始)
  - [安装](#安装)
  - [启动 Web 工作台](#启动-web-工作台)
  - [CLI 使用](#cli-使用)
- [项目结构](#项目结构)
- [发布新版本](#发布新版本)
- [许可证](#许可证)

---

## 环境准备

- **Python** 3.10+
- **Node.js** 20（仅前端开发/构建需要）
- CUDA 环境（GPU 训练时）
- Windows / Linux / macOS 均可

---

## 快速开始

### 安装

```bash
# 1. 克隆仓库
cd EdgeDistillDet

# 2. Windows 推荐一键安装（自动安装/升级 Python 3.10+，并匹配 CUDA 通道）
installer.bat

# 3. 手动安装（跨平台）
pip install -r requirements.txt

# 验证安装
edgedistilldet --help
```

### 启动 Web 工作台

```bash
# 构建前端（首次或前端代码更新后）
cd web && npm ci && npm run build && cd ..

# 启动服务
python web/app.py
```

访问 http://127.0.0.1:5000

**开发模式**（前后端热更新）：
- 终端一：`python web/app.py`（后端）
- 终端二：`cd web && npm run dev`（前端，端口 5173）

### CLI 使用

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

## 项目结构

```
EdgeDistillDet/
├── main.py                 # CLI 统一入口
├── core/
│   ├── distillation/         # 蒸馏训练核心
│   ├── evaluation/           # benchmark 评估
│   └── detection/            # 推理/检测封装
├── scripts/                  # 训练/评估/分析脚本
├── utils/                    # 数据分析 / 边缘剖析 / 可视化
├── web/
│   ├── app.py                # FastAPI 主入口
│   ├── routers/              # API 路由
│   ├── services/             # 业务逻辑
│   ├── src/                  # React 前端源码
│   └── static/dist/          # 前端构建产物
├── configs/                  # 训练/评估/数据集配置
├── tests/                    # pytest 测试
├── pyproject.toml            # 项目依赖与元数据
└── requirements.txt          # pip 安装入口
```

### 关键配置文件

| 文件 | 用途 |
|------|------|
| `configs/distill_config.yaml` | 教师/学生模型、蒸馏参数、训练超参、输出目录 |
| `configs/eval_config.yaml` | 待评估权重列表、数据集、设备 batch 大小 |
| `docs/PARAMETER_MAPPING.md` | 前后端参数映射矩阵（含高级参数与续训锁定字段） |

---

## 发布新版本

1. **改版本号**：修改 `main.py` 中的 `__version__`（`pyproject.toml` 会自动读取）
2. **同步前端版本**：修改 `web/package.json` 和 `web/package-lock.json` 中的 `version`
3. **写变更记录**：在 `docs/CHANGELOG.md` 添加 `## [x.y.z] - YYYY-MM-DD`
4. **构建前端**：`cd web && npm ci && npm run build`，提交 `web/static/dist/`
5. **自检**：`python -c "from main import __version__; print(__version__)"` 和 `GET /api/version`

---

## 许可证

[MIT](LICENSE)
