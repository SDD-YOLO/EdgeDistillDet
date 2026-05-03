# 模型蒸馏全流程管理平台 - 开发任务清单

> 目标：构建一个支持本地训练、云端提交、实验管理和智能分析的模型蒸馏平台。

[x]补全断点续练的不可以切换设备和配置的参数的逻辑
[x]更改logo，并且让logo支持明暗切换
[x]将task: 任务类型 (detect/segment)mode: 运行模式 (train/val/predict)model: 学生模型权重路径data: 数据集 yaml 路径epochs: 总训练轮数time: 限时训练时长patience: 早停轮数batch: 批次大小imgsz: 图像尺寸save: 是否保存模型save_period: 间隔轮数保存cache: 数据集缓存 (false/ram/disk)device: 运行设备 (cpu/0/1)workers: 数据加载线程project: 结果保存根目录name: 实验文件夹名exist_ok: 覆盖同名文件夹pretrained: 是否使用预训练权重optimizer: 优化器类型 (auto/SGD/Adam)verbose: 输出详细日志seed: 随机种子deterministic: 固定随机因子single_cls: 是否单类别训练rect: 矩形训练cos_lr: 余弦学习率close_mosaic: 关闭马赛克增强轮数resume: 断点续训amp: 混合精度训练fraction: 数据集使用比例profile: 性能分析freeze: 冻结模型层数multi_scale: 多尺度训练compile: 模型编译加速overlap_mask: 掩码重叠mask_ratio: 掩码下采样dropout: 随机失活val: 训练时验证split: 验证集划分save_json: 保存验证 jsonconf: 置信度阈值iou: NMS 阈值max_det: 单图最大检测数half: 半精度dnn: OpenCV 推理plots: 生成训练图表end2end: 端到端模式source: 推理数据源vid_stride: 视频步长stream_buffer: 流缓冲visualize: 特征可视化augment: 推理增强agnostic_nms: 类别无关 NMSclasses: 指定检测类别retina_masks: 高分辨率掩码embed: 特征导出show: 实时预览save_frames: 保存帧save_txt: 保存标签save_conf: 保存置信度save_crop: 保存裁剪结果show_labels: 显示标签show_conf: 显示置信度show_boxes: 显示检测框line_width: 框线宽度format: 模型导出格式keras: Keras 导出optimize: 模型优化int8: INT8 量化dynamic: 动态输入simplify: 简化模型opset: ONNX 版本workspace: 工作空间nms: 启用 NMSlr0: 初始学习率lrf: 最终学习率系数momentum: 优化器动量weight_decay: 权重衰减warmup_epochs: 预热轮数warmup_momentum: 预热动量warmup_bias_lr: 偏置预热学习率box: 检测框损失权重cls: 分类损失权重cls_pw: 分类损失平衡系数dfl: 分布焦点损失权重pose: 姿态损失权重kobj: 关键点损失权重rle: 掩码编码系数angle: 旋转角度系数nbs: 标准化批次hsv_h: 色相增强hsv_s: 饱和度增强hsv_v: 亮度增强degrees: 旋转角度translate: 平移幅度scale: 缩放幅度shear: 错切变换perspective: 透视变换flipud: 上下翻转概率fliplr: 左右翻转概率bgr: BGR 通道转换mosaic: 马赛克增强mixup: 混合增强cutmix: 剪切混合copy_paste: 复制粘贴增强copy_paste_mode: 增强模式auto_augment: 自动增强策略erasing: 随机擦除cfg: 模型配置文件tracker: 追踪器配置save_dir: 自动生成的保存路径teacher_weights: 教师模型权重路径 teacher_cfg: 教师模型配置文件distill_mode: 蒸馏模式 (adaptive 自适应 /feature 特征 /response 响应)
temperature: 蒸馏温度 (推荐 2~8)alpha: 软标签损失权重 (推荐 0.3~0.8)beta: 特征蒸馏损失权重 (推荐 0.001~0.01)gamma: 关系蒸馏损失权重cls_distill_weight: 分类分支蒸馏权重box_distill_weight: 检测框分支蒸馏权重obj_distill_weight: 置信度分支蒸馏权重distill_loss_type: 损失类型 (kl/mse/l1) freeze_teacher: 冻结教师模型 (True/False)teacher_device: 教师模型运行设备distill_start_epoch: 蒸馏开始轮数distill_end_epoch: 蒸馏结束轮数dynamic_alpha: 动态权重调度开关dynamic_temperature: 动态温度调度开关 feature_layers: 特征蒸馏层选择lambda_kd: 总蒸馏损失系数use_adaptive_loss: 是否启用自适应损失

这些可以修改的参数直接全部在前端添加一个新的“高级参数配置”展示出来，确保所有的前端的参数设置窗口和后端是正确的映射关系
[x]下载依赖的CUDA版本的时候先查看一下设备的配置并下载对应的CUDA版本，确保下载的CUDA版本和设备配置是匹配的
[x]仔细检查代码中有没有硬编码，如果有的话需要修改
[x]仔细检查代码中有没有重复的代码，如果有的话需要合并
[x]仔细检查代码中有没有潜在的bug，如果有的话需要修复
[x]补全模块测试，将项目中的所有的代码的模块化进行到底，确保代码的最高可复用程度
[x]installer脚本中添加一个如果没有python就下载python的逻辑，如果有python就升级python的逻辑，如果有python但是版本不符合就升级python的逻辑，确保安装的python版本是3.10+

[x]前端的指标因为过滤的规则过于严格而无法显示的bug

- 修复: web/src/features/metrics/MetricsPanel.jsx 中的 renderAllCharts 函数
- 改进: 移除了 minPositiveLen 的严格限制，改为使用 padToLength 补充缺失数据
- 结果: 即使某个指标缺失，其他指标仍可正常显示

[x]高级参数设置的界面中有一些参数和既定的流程冲突或者冗余

- 修复: web/src/features/training/TrainingPanel.jsx 中的 renderAdvancedField 函数
- 改进: 对 resume 模式应用更精细的参数限制
- 添加: RESUME_UNLOCKED_PARAMS 白名单，允许在 resume 模式下修改特定参数（如 time, patience 等）
- 添加: tooltip 提示用户哪些参数在 resume 模式下被禁用

[x]后端数据链路的 4 个关键缺陷
[x]1. results.csv 扫描路径硬编码 - 修复: web/services/backend_metrics.py 中添加 \_get_candidate_runs_directories() 函数 - 改进: 支持多路径扫描（runs/, runs/detect/, 自定义目录等）- 支持: EDGE_RUNS_DIRS 环境变量指定自定义扫描路径 - 结果: 训练结果不再受硬编码路径限制

[x]2. 蒸馏指标列与 Ultralytics 存在"覆写竞争" - 修复: core/distillation/adaptive_kd_trainer.py 中的 \_on_fit_epoch_end 方法 - 改进: 添加 \_save_distill_log_json() 备用持久化机制 - 改进: 增强 \_on_train_end 中的安全网逻辑，更好地处理异常中断情况 - 结果: 蒸馏指标即使 Ultralytics 覆写 CSV 也能恢复

[x]3. distill_log.json 回退机制失效 - 修复: web/services/backend_common.py 中的 \_build_metric_series 函数 - 改进: 改进了 distill_log_fallback 的加载逻辑，只在真正缺失数据时才加载 - 改进: 允许数据为 None，而不是强制转换为 0 - 结果: 蒸馏数据可靠地从备用源恢复

[x]4. CSV 列名严格匹配，版本兼容性差 - 修复: web/services/backend_common.py 中添加列名动态探测机制 - 改进: 添加 \_resolve_column_name() 和 \_METRIC_COLUMN_ALIASES 映射表 - 支持: 多个 YOLO 版本的列名变体（如 mAP50 vs mAP50(B)）- 结果: 自动适应不同 Ultralytics 版本的列名差异

[x]TrainingPanel 拆解模块

- 创建: 4 个专用自定义 Hooks
  - useTrainingState - 训练状态管理
  - useExportState - 导出状态管理
  - useInferenceState - 推理状态管理
  - useResumeState - 断点续训状态管理
- 创建: web/src/features/training/hooks/ 目录
- 创建: REFACTORING_GUIDE.md 详细优化指南
- 结果: 为 TrainingPanel 进一步拆解奠定基础，遵循单一职责原则

[x]TraingingPanel拆解模块 - 创建: 4 个视图容器组件
_ TrainingViewContainer - 训练页主布局
_ ExportViewContainer - 导出页容器
_ DisplayViewContainer - 推理页容器
_ AdvancedViewContainer - 高级参数页容器 - 结果: TrainingPanel 的 JSX 视图分支已从主文件中抽离，便于继续分层拆解

EdgeDistillDet 底层框架优化分析报告

涵盖范围: 后端架构、API 层、核心训练模块、前端工程化、依赖管理、安全运维

---

一、指标监控显示为 0 的问题分析与修复

1.1 根因定位

问题出在 `web/services/backend_common.py` 的 `_build_metric_series` 函数中：指标数据提取使用了硬编码的 Ultralytics YOLOv8 默认列名，且对空值使用 `or 0` 回退。

```python
# 硬编码列名 + or 0 回退（原代码）
chart['map_series']['map50'].append(_as_float(row.get('metrics/mAP50(B)')) or 0)
chart['map_series']['map50_95'].append(_as_float(row.get('metrics/mAP50-95(B)')) or 0)
chart['precision_recall']['precision'].append(_as_float(row.get('metrics/precision(B)')) or 0)
```

1.2 具体原因

场景 说明
YOLO 版本差异 YOLOv6/v9 等版本的 `results.csv` 列名可能不带 `(B)` 后缀，如 `metrics/mAP50`
列名前缀差异 某些版本使用 `val/box_loss` 而非 `train/box_loss`
`or 0` 回退 列名不匹配时，`row.get()` 返回 `None`，`or 0` 使其变成 `0`
后端 summary 同样受影响 `_summarize_series` 中也使用相同硬编码列名

1.3 不同 YOLO 版本的列名对比

指标 YOLOv8 格式 YOLOv6/v9 格式
mAP50 `metrics/mAP50(B)` `metrics/mAP50`
mAP50-95 `metrics/mAP50-95(B)` `metrics/mAP50-95`
Precision `metrics/precision(B)` `metrics/precision`
Recall `metrics/recall(B)` `metrics/recall`
Box Loss `train/box_loss` `box_loss`

1.4 修复方案

1.4.1 添加列名自动探测与兼容映射

在 `web/services/backend_common.py` 中添加列名解析辅助函数和映射表：

```python
def _resolve_column_name(columns, candidates):
    """从实际 CSV 列名中匹配候选列名"""
    if not columns:
        return None
    col_map = {col.lower().strip(): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        lookup = candidate.lower().strip()
        if lookup in col_map:
            return col_map[lookup]
    return None

_METRIC_COLUMN_ALIASES = {
    'box_loss': ['train/box_loss', 'box_loss', 'train_box_loss'],
    'cls_loss': ['train/cls_loss', 'cls_loss', 'train_cls_loss'],
    'dfl_loss': ['train/dfl_loss', 'dfl_loss', 'train_dfl_loss'],
    'map50': ['metrics/mAP50(B)', 'metrics/mAP50', 'mAP50', 'map50'],
    'map50_95': ['metrics/mAP50-95(B)', 'metrics/mAP50-95', 'mAP50-95', 'map50_95'],
    'precision': ['metrics/precision(B)', 'metrics/precision', 'precision'],
    'recall': ['metrics/recall(B)', 'metrics/recall', 'recall'],
    'lr_pg0': ['lr/pg0', 'lr_pg0'],
    'lr_pg1': ['lr/pg1', 'lr_pg1'],
    'lr_pg2': ['lr/pg2', 'lr_pg2'],
}
```

1.4.2 重写 `_build_metric_series`

使用动态列名探测替代硬编码：

```python
def _build_metric_series(rows, columns, run_dir):
    col = {}
    for metric_key, aliases in _METRIC_COLUMN_ALIASES.items():
        col[metric_key] = _resolve_column_name(columns, aliases)

    for row in rows:
        epoch = _as_float(row.get('epoch'))
        if epoch is None:
            continue
        # 动态提取指标，列不存在时保留 None
        m50_val = _as_float(row.get(col['map50'])) if col['map50'] else None
        # ... 其他指标同理
```

1.4.3 同步修复 `backend_metrics.py` 中的 summary 指标

```python
_summary_metric_map = [
    ('box_loss', ['train/box_loss', 'box_loss', 'train_box_loss'], 'lower'),
    ('map50', ['metrics/mAP50(B)', 'metrics/mAP50', 'mAP50', 'map50'], 'higher'),
    # ...
]
for key, aliases, better in _summary_metric_map:
    actual_col = _resolve_column_name(columns, aliases)
    if not actual_col:
        continue
```

---

二、架构层优化

2.1 引入状态持久化层（最高优先级）

现状：所有状态通过文件系统管理，每次 API 请求都重新 `os.walk` 扫描并解析 CSV。

问题：

- 高并发时大量磁盘 I/O
- 没有历史数据的增量更新机制
- 训练状态依赖文件系统，不可靠

优化方案：

- 引入 SQLite 或 TinyDB 作为元数据存储层
- 训练运行时写入数据库，Web 端从数据库读取
- 文件系统保留为原始数据源，数据库作为查询缓存层
- 可选：使用 SQLAlchemy + Alembic 做 ORM 和迁移管理

  2.2 引入任务队列替代子进程管理

现状：`backend_train_runtime.py` 使用 `subprocess.Popen` + 文件锁管理训练进程。

优化方案：

- 使用 Celery + Redis 或 RQ 将训练任务异步化
- 好处：天然支持任务状态追踪、重试、并发控制、结果回调
- WebSocket 推送训练进度时，从任务队列消费事件而非轮询文件

  2.3 拆分 God Class

现状：`backend_common.py` 636 行，混合了 CSV 解析、YAML 读写、路径扫描、模型估算、checkpoint 管理等十几种职责。

建议拆分结构：

```
web/services/
├── io/               # 文件读写（csv, yaml, json）
├── scan/             # 目录扫描、checkpoint 发现
├── metrics/          # 指标计算与汇总
├── profile/          # 模型参数量/GFLOPs 估算
├── config/           # 配置验证与解析
└── cache/            # 缓存层
```

---

三、API 层优化

3.1 统一配置管理（Pydantic Settings）

现状：`settings.py` 手工读取环境变量，无类型验证、无默认值提示。

优化方案：

```python
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 5000
    cors_origins: list[str] = []
    db_path: str = "./data/app.db"

    class Config:
        env_prefix = "EDGE_"
        env_file = ".env"

settings = AppSettings()
```

3.2 增加 API 基础设施

现状缺失：

- 统一响应格式
- 全局异常处理中间件
- 请求日志/追踪
- API 认证
- 限流（Rate Limiting）

优化方案：

```python
# 统一响应模型
class ResponseModel[T](BaseModel):
    code: int = 0
    data: T | None = None
    message: str = "ok"

# 全局异常处理
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc): ...

# 限流
from slowapi import Limiter
limiter = Limiter(key_func=lambda: "global")
```

3.3 WebSocket 与 FastAPI 原生集成

现状：`websocket_server.py` 是独立模块，使用自定义 `ConnectionManager` + `threading.Lock`。

优化方案：

- 使用 FastAPI 原生 `@app.websocket("/ws/train")`
- 用 `asyncio.Queue` 替代线程锁做消息广播
- 集成到 `app.py` 中，无需单独运行

---

四、核心训练层优化

4.1 指标采集管道化

现状：训练指标通过 Ultralytics 的 `results.csv` 被动写入，Web 端主动轮询文件。

优化方案：

实现 Ultralytics 回调系统直接推送指标：

```python
# core/distillation/callbacks.py
class MetricsCallback:
    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch
        metrics = trainer.metrics
        # 直接写入 DB 或推送到 WebSocket
        push_metrics(epoch, metrics)
```

4.2 模型版本管理

现状：`yolo26n.pt` 直接放在仓库根目录，Git 管理二进制权重效率极低。

优化方案：

- 使用 Git LFS 或从 HuggingFace 按需下载
- 提供 `scripts/download_models.py` 脚本
- 在 `.gitignore` 中排除 `*.pt`

---

五、前端层优化

5.1 状态管理升级

现状：`MetricsPanel.jsx` 733 行，组件内部状态过多。

优化方案：

- 引入轻量级状态管理库（Zustand 或 Jotai）
- 将 metrics、training、config 等状态拆分为独立 store

  5.2 API 客户端增强

优化方案：

- 使用 TanStack Query (React Query) 替代手写 `useEffect` 轮询
- 增加请求拦截器、重试机制、请求取消

  5.3 前端性能优化

- 使用 `useMemo` / `useCallback` 避免重复渲染
- Chart.js 数据量大时考虑虚拟化或采样
- 使用 `React.lazy` 做路由级代码分割

---

六、依赖与工程化优化

6.1 依赖瘦身

现状：同时依赖 PyTorch、TensorFlow、ONNXRuntime、OpenVINO、MNN、NCNN、CoreML，体积巨大。

优化方案：按功能拆分 extras：

```toml
[project.optional-dependencies]
tf = ["tensorflow>=2.16.0"]
edge = ["openvino>=2024.0.0", "nncf>=2.19.0", "MNN>=3.0.0"]
ml = ["coremltools>=8.0.0"]
dev = ["pytest", "black", "ruff", "mypy"]
```

6.2 代码质量工具链

工具 用途
Ruff 替代 flake8 + black，极速 Lint 和格式化
mypy 严格类型检查
pre-commit 提交前自动格式化和检查
pytest + coverage 测试框架和覆盖率

6.3 日志系统结构化

现状：大量使用 `print()` 和普通 `logging`，格式不统一。

优化方案：

```python
import structlog

logger = structlog.get_logger()
logger.info("training_started", epoch=10, loss=0.5, gpu="cuda:0")
# 输出 JSON 格式，便于 ELK/Loki 收集
```

---

七、安全与运维优化

7.1 CORS 与安全性

现状：`allow_methods=["*"], allow_headers=["*"]` 过于宽松。

优化：显式声明允许的方法和头，生产环境关闭 `*`。

7.2 健康检查与监控

新增 `/health` 端点：

```python
@app.get("/health")
def health():
    return {
        "status": "ok",
        "db": check_db(),
        "gpu": check_gpu(),
        "queue": check_task_queue()
    }
```

7.3 容器化

提供 `Dockerfile` 和 `docker-compose.yml`：

```yaml
services:
  app:
    build: .
    ports: ["5000:5000"]
    volumes:
      - ./runs:/app/runs
      - ./data:/app/data
  redis:
    image: redis:alpine
```

---

八、优化路线图

阶段 优先级 内容 预计投入
P0 立即 指标列名动态探测修复、统一错误处理、拆分 `backend_common.py` 1-2 天
P1 短期 引入 SQLite 元数据缓存、Celery 任务队列、Pydantic Settings 重构 1-2 周
P2 中期 FastAPI WebSocket 原生集成、前端 TanStack Query + Zustand、依赖瘦身 2-3 周
P3 长期 Docker 容器化、结构化日志、完整测试覆盖、监控告警体系 1 个月+

---

附录：涉及修改的文件清单

文件路径 修改类型 说明
`web/services/backend_common.py` 重构 添加列名探测、拆分职责
`web/services/backend_metrics.py` 修改 summary 指标动态列名
`web/services/metrics_service.py` 可选 接入数据库缓存层
`web/app.py` 修改 集成 WebSocket、增加中间件
`web/core/settings.py` 重构 Pydantic Settings
`web/websocket_server.py` 重构 迁移到 FastAPI 原生 WebSocket
`pyproject.toml` 修改 拆分 optional-dependencies
`core/distillation/` 新增 训练回调推送指标
`web/src/features/metrics/` 优化 前端状态管理和 API 调用
