# EdgeDistillDet

面向边缘场景的微小目标蒸馏训练与评估项目，包含：

- 训练与蒸馏（`Ultralytics + AdaptiveKDTrainer`）
- 统一 CLI（训练/评估/数据分析/边缘剖析）
- 本地 Web 工作台（FastAPI + React）
- 配置驱动的实验流程（`configs/*.yaml`）

**版本**：发布版本在 [`main.py`](main.py) 的 `__version__` 中维护（与 `pip install` 后的包版本一致）。查看：`python -c "from main import __version__; print(__version__)"` 或 `python -m pip show edgedistilldet`；Web 服务：`GET /api/version`。变更记录见 [CHANGELOG.md](CHANGELOG.md)。**许可证**：[MIT](LICENSE)。

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
├─ tests/                          # pytest 关键路径测试
├─ pyproject.toml                  # 项目元数据与 Python 依赖（PEP 621）
├─ LICENSE                         # MIT
├─ CHANGELOG.md                    # 版本变更记录
├─ requirements.txt                # 唯一 pip 入口（可编辑安装；依赖见 pyproject.toml）
└─ docs/regression_baseline.md     # 重构前行为快照
```

## 2. 环境要求

- Python 3.10+（建议 Conda）
- Node.js：**推荐 Node 20**（与仓库根目录 `.nvmrc` 及 GitHub Actions CI 一致）；`web/package.json` 的 `engines` 为 `>=18`，更低主版本需自行验证。仅 Web 前端开发/构建需要安装 Node。
- Windows/Linux/macOS 均可
- GPU 训练建议 CUDA 环境可用

## 3. 快速开始

### 3.1 安装 Python 依赖

运行依赖在 **`pyproject.toml`** 的 `[project.dependencies]` 与根目录 **`requirements.txt`** 中列出（两处需保持同步）；推荐从仓库根目录安装：

```bash
pip install -r requirements.txt
```

会安装所列第三方包并以可编辑方式安装本项目（末尾 `-e .`）。安装后可在任意目录使用 **`edgedistilldet`** 命令，等价于 `python main.py`。

运行自动化测试（可选）：

```bash
pip install -e ".[dev]"
python -m pytest
```

`pip install -e ".[dev]"` 会安装 `pytest` 等可选开发依赖（定义见 `pyproject.toml` 的 `[project.optional-dependencies].dev`）。

### 3.2 构建前端（生产静态资源）

在 **`web/`** 目录安装与构建（与 `package-lock.json` 一致，请使用 `npm ci` 而非 `npm install`，避免锁文件漂移）：

```bash
cd web
npm ci
npm run build
```

构建输出目录：`web/static/dist/`（`app.js` 与 `app.css`）

**与仓库中 `dist` 的一致性（CI）**：`web/static/dist/` 中的文件**需要提交**。GitHub Actions 会在每次 push / PR 时在 Ubuntu 上执行 `npm ci && npm run build`，并校验工作区中 `web/static/dist/` 与构建结果一致；不一致时 CI 失败。建议在仓库根目录使用 **Node 20**（可用 `.nvmrc`），在 `web/` 下执行 `npm ci && npm run build` 后再提交。

### 3.3 启动 Web（FastAPI）

```bash
python web/app.py
```

默认仅监听本机 **`127.0.0.1:5000`**（非全网卡暴露）。浏览器访问：`http://127.0.0.1:5000`

### 3.4 前端开发模式（Vite）

终端一：启动后端（同上）。终端二：

```bash
cd web
npm ci
npm run dev
```

开发服务器默认 `http://127.0.0.1:5173`，已在 `vite.config.js` 中将 `/api`、`/static`、`/favicon.ico` **代理到** `http://127.0.0.1:5000`，前端代码使用相对路径 `/api/...` 即可，无需额外配置 CORS。若后端端口不是 `5000`，请同步修改 `web/vite.config.js` 中的 `server.proxy` 目标地址。

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

### 6.1 监听地址与 CORS（安全说明）

| 环境变量 | 含义 | 默认 |
|----------|------|------|
| `EDGE_BACKEND_HOST` | 绑定地址 | `127.0.0.1`（仅本机） |
| `EDGE_BACKEND_PORT` | 端口（兼容旧名 `EDGE_FLASK_PORT`） | `5000` |
| `EDGE_CORS_ORIGINS` | 允许的浏览器来源，逗号分隔；设为单个 `*` 则任意来源（**不**携带 `credentials`） | 本机 `5000/5173` 的常用 origin |

局域网内其他机器访问时，可将 `EDGE_BACKEND_HOST=0.0.0.0`，**请在可信网络使用**；本机独占使用时保持默认即可。

### 6.2 接口列表（与路由实现一致）

**元数据**

- `GET /api/version`：返回 `{"name":"edgedistilldet","version":"..."}`（版本与 `main.__version__` 一致；同时体现在 OpenAPI 文档的 API 版本字段中）

**配置与对话框**（[`web/routers/config.py`](web/routers/config.py)）

- `GET /api/configs`：列出可用配置
- `GET /api/config/{config_name}`：读取指定配置
- `GET /api/config/recent`：最近使用的配置
- `POST /api/config/save`：保存配置
- `POST /api/config/upload`：上传配置
- `POST /api/dialog/pick`：路径选择对话框（服务端交互）

**训练与输出**（[`web/routers/train.py`](web/routers/train.py)）

- `GET /api/output/check`：检查输出目录（查询参数 `project` 等）
- `POST /api/train/start`：启动训练
- `POST /api/train/stop`：停止训练
- `GET /api/train/status`：训练状态
- `GET /api/train/resume_candidates`：可选续训 checkpoint 列表
- `GET /api/train/logs`：训练日志分页
- `GET /api/train/logs/download`：下载日志
- `GET /api/train/logs/stream`：日志流式输出

**指标**（[`web/routers/metrics.py`](web/routers/metrics.py)）

- `GET /api/metrics`：训练指标

**Agent**（[`web/routers/agent.py`](web/routers/agent.py)）

- `GET /api/agent/config-schema`：配置 JSON Schema
- `POST /api/agent/patch/validate`：校验补丁
- `POST /api/agent/patch/preview`：预览补丁
- `POST /api/agent/patch/apply`：应用补丁
- `GET /api/agent/run/{run_id}/history`：某次运行的配置历史
- `POST /api/agent/run/{run_id}/rollback`：按版本回滚配置
- `GET /api/agent/tools`：工具契约
- `POST /api/agent/tools/execute`：执行工具
- `POST /api/agent/model/invoke`：模型调用
- `POST /api/agent/model/invoke-stream`：模型流式调用

## 7. 常见问题

### 7.1 页面空白或静态资源 404

- 确认已执行 `npm run build`
- 确认存在 `web/static/dist/app.js` 和 `web/static/dist/app.css`

### 7.2 训练无法启动

- 检查是否已有残留训练进程
- 检查 `.training.lock` 是否被占用
- 检查 GPU 显存是否不足

### 7.3 评估找不到数据集

- 检查 `eval_config.yaml` 中 `test_yaml` 路径
- 相对路径默认相对于配置文件所在目录与仓库根目录进行解析

### 7.4 日志显示异常或刷新慢

- Web 启动训练时会启用 `EDGE_WEB_LOG=1` 改善管道日志可读性
- 在 Windows 下建议使用项目默认启动方式，避免额外重定向

## 8. 开发者说明

### 8.1 模块化约定

为提升可维护性，代码组织遵循：

- Web 请求模型统一放在 `web/schemas.py`
- Web 业务逻辑优先放到 `web/services/*`
- 共享模型统计逻辑统一放在 `core/model_metrics.py`
- 数据集共享逻辑统一放在 `utils/dataset_common.py`
- 前端通用能力按职责拆分：`web/src/api/client.js`、`web/src/hooks/useToast.js`、`web/src/constants/trainingDefaults.js`、`web/src/utils/logging.js`、`web/src/utils/time.js`

### 8.2 回归验证建议

1. `python main.py --help` 正常显示四个子命令
2. `pip install -e ".[dev]"` 后执行 `python -m pytest` 通过
3. Web 启动后访问：`GET /api/version`、`GET /api/configs`、`GET /api/train/status`、`GET /api/metrics`
4. 对比 [`docs/regression_baseline.md`](docs/regression_baseline.md) 的字段结构与语义是否一致

### 8.3 备注

本次结构重构遵循「**不改功能语义**」原则：路径、输入输出字段、训练主流程保持兼容，重点提升模块边界、复用度与可读性。

## 9. 发布新版本

发版前建议逐项核对：

1. **Python 包版本**：仅修改 [`main.py`](main.py) 中的 `__version__`（[`pyproject.toml`](pyproject.toml) 通过 `dynamic` 从该属性读取，勿手写重复版本号）。
2. **Web UI 包版本**：同步修改 [`web/package.json`](web/package.json) 的 `version`；若变更了顶层 `version` 字段，检查 [`web/package-lock.json`](web/package-lock.json) 中根包 `"packages": {""}` 的 `version` 是否一致。
3. **记录变更**：在 [CHANGELOG.md](CHANGELOG.md) 增加 `## [x.y.z] - YYYY-MM-DD` 小节。
4. **前端构建产物**：在 `web/` 下执行 `npm ci && npm run build`，确认 [`web/static/dist/`](web/static/dist/) 与 CI 预期一致后再提交。
5. **快速自检**：`python -c "from main import __version__; print(__version__)"`、`python -m pip show edgedistilldet`、`GET /api/version`。
