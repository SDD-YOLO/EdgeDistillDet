const boolOptions = [
  { value: "true", label: "启用" },
  { value: "false", label: "禁用" }
];

const computeProviderOptions = [
  { value: "local", label: "本地" },
  { value: "autodl", label: "AutoDL" },
  { value: "colab", label: "Colab" },
  { value: "remote_api", label: "远程 API" }
];

const deviceOptions = [
  { value: "0", label: "GPU 0" },
  { value: "0,1", label: "GPU 0,1" },
  { value: "cpu", label: "CPU" }
];

const optimizerOptions = [
  { value: "auto", label: "自动 (Auto)" },
  { value: "SGD", label: "SGD" },
  { value: "Adam", label: "Adam" },
  { value: "AdamW", label: "AdamW" }
];

const cacheOptions = [
  { value: "false", label: "禁用" },
  { value: "ram", label: "RAM" },
  { value: "disk", label: "磁盘" }
];

const taskOptions = [
  { value: "detect", label: "detect" },
  { value: "segment", label: "segment" },
  { value: "pose", label: "pose" },
  { value: "classify", label: "classify" }
];

const runModeOptions = [
  { value: "train", label: "train" },
  { value: "val", label: "val" },
  { value: "predict", label: "predict" }
];

const distillModeOptions = [
  { value: "adaptive", label: "自适应蒸馏" },
  { value: "feature", label: "特征蒸馏" },
  { value: "response", label: "响应蒸馏" }
];

const distillLossTypeOptions = [
  { value: "kl", label: "KL" },
  { value: "mse", label: "MSE" },
  { value: "l1", label: "L1" }
];

const exportFormatOptions = [
  { value: "onnx", label: "ONNX" },
  { value: "torchscript", label: "TorchScript" }
];

const wandbModeOptions = [
  { value: "online", label: "online" },
  { value: "offline", label: "offline" },
  { value: "disabled", label: "disabled" }
];

const imageVideoFilters = [
  { name: "媒体文件", patterns: ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.mp4", "*.avi", "*.mov", "*.mkv"] },
  { name: "所有文件", patterns: ["*.*"] }
];

const weightFilters = [
  { name: "PyTorch 权重", patterns: ["*.pt", "*.pth"] },
  { name: "所有文件", patterns: ["*.*"] }
];

const yamlFilters = [
  { name: "YAML 文件", patterns: ["*.yaml", "*.yml"] },
  { name: "所有文件", patterns: ["*.*"] }
];

const directoryFilters = [{ name: "所有文件夹", patterns: ["*.*"] }];

const distillationParams = [
  { key: "student_weight", path: "distillation.student_weight", label: "学生模型权重", type: "path", required: true, browseKind: "file", filters: weightFilters, placeholder: "models/yolov8n.pt", hint: "学生模型权重文件" },
  { key: "teacher_weight", path: "distillation.teacher_weight", label: "教师模型权重", type: "path", required: true, browseKind: "file", filters: weightFilters, placeholder: "models/yolov8m.pt", hint: "教师模型权重文件" },
  { key: "alpha_init", path: "distillation.alpha_init", label: "蒸馏初始权重 α", type: "number", min: 0, max: 1, step: 0.01, required: true, hint: "蒸馏损失在总损失中的初始占比" },
  { key: "T_max", path: "distillation.T_max", label: "最大温度 T_max", type: "number", min: 0.1, step: 0.1, hint: "软标签温度上限" },
  { key: "T_min", path: "distillation.T_min", label: "最小温度 T_min", type: "number", min: 0.1, step: 0.1, hint: "软标签温度下限" },
  { key: "warm_epochs", path: "distillation.warm_epochs", label: "蒸馏预热轮数", type: "number", min: 0, step: 1, hint: "纯监督训练的预热轮数" },
  { key: "w_kd", path: "distillation.w_kd", label: "KD 权重 w_kd", type: "number", min: 0, step: 0.1 },
  { key: "w_focal", path: "distillation.w_focal", label: "Focal 权重 w_focal", type: "number", min: 0, step: 0.1 },
  { key: "w_feat", path: "distillation.w_feat", label: "特征权重 w_feat", type: "number", min: 0, step: 0.1 },
  { key: "scale_boost", path: "distillation.scale_boost", label: "尺度增强 scale_boost", type: "number", min: 0, step: 0.1 },
  { key: "focal_gamma", path: "distillation.focal_gamma", label: "聚焦系数 focal_gamma", type: "number", min: 0, step: 0.1 },
  { key: "distill_mode", path: "distillation.distill_mode", label: "蒸馏模式", type: "enum", options: distillModeOptions },
  { key: "distill_start_epoch", path: "distillation.distill_start_epoch", label: "蒸馏起始 Epoch", type: "number", min: 0, step: 1 },
  { key: "distill_end_epoch", path: "distillation.distill_end_epoch", label: "蒸馏结束 Epoch", type: "number", min: 0, step: 1 },
  { key: "freeze_teacher", path: "distillation.freeze_teacher", label: "冻结教师", type: "enum", options: boolOptions },
  { key: "dynamic_alpha", path: "distillation.dynamic_alpha", label: "动态 α", type: "enum", options: boolOptions },
  { key: "dynamic_temperature", path: "distillation.dynamic_temperature", label: "动态温度", type: "enum", options: boolOptions },
  { key: "cls_distill_weight", path: "distillation.cls_distill_weight", label: "分类蒸馏权重", type: "number", min: 0, step: 0.1 },
  { key: "box_distill_weight", path: "distillation.box_distill_weight", label: "框蒸馏权重", type: "number", min: 0, step: 0.1 },
  { key: "obj_distill_weight", path: "distillation.obj_distill_weight", label: "目标蒸馏权重", type: "number", min: 0, step: 0.1 },
  { key: "distill_loss_type", path: "distillation.distill_loss_type", label: "蒸馏损失类型", type: "enum", options: distillLossTypeOptions },
  { key: "teacher_cfg", path: "distillation.teacher_cfg", label: "教师模型配置", type: "text" },
  { key: "teacher_device", path: "distillation.teacher_device", label: "教师设备", type: "text" },
  { key: "feature_layers", path: "distillation.feature_layers", label: "特征层", type: "text" },
  { key: "lambda_kd", path: "distillation.lambda_kd", label: "lambda_kd", type: "number", min: 0, step: 0.1 },
  { key: "use_adaptive_loss", path: "distillation.use_adaptive_loss", label: "自适应损失", type: "enum", options: boolOptions },
  { key: "temperature", path: "distillation.temperature", label: "temperature", type: "number", min: 0, step: 0.1 },
  { key: "alpha", path: "distillation.alpha", label: "alpha", type: "number", min: 0, step: 0.1 },
  { key: "beta", path: "distillation.beta", label: "beta", type: "number", min: 0, step: 0.1 },
  { key: "gamma", path: "distillation.gamma", label: "gamma", type: "number", min: 0, step: 0.1 }
];

const trainingCoreParams = [
  { key: "compute_provider", path: "training.compute_provider", label: "计算环境", type: "enum", options: computeProviderOptions, required: true },
  { key: "data_yaml", path: "training.data_yaml", label: "数据集配置", type: "path", required: true, browseKind: "file", filters: yamlFilters, placeholder: "configs/dataset_coco128.yaml", hint: "YOLO 数据集 YAML 路径" },
  { key: "device", path: "training.device", label: "训练设备", type: "enum", options: deviceOptions, required: true },
  { key: "epochs", path: "training.epochs", label: "训练轮数", type: "number", min: 1, max: 1000, step: 1, required: true },
  { key: "imgsz", path: "training.imgsz", label: "输入尺寸", type: "number", min: 32, max: 2048, step: 32, required: true },
  { key: "batch", path: "training.batch", label: "批次大小", type: "number", min: -1, step: 1, hint: "-1 表示自动批次" },
  { key: "lr0", path: "training.lr0", label: "初始学习率", type: "number", min: 0, step: 0.0001 },
  { key: "lrf", path: "training.lrf", label: "最终学习率系数", type: "number", min: 0, step: 0.01 },
  { key: "max_det", path: "training.max_det", label: "最大检测数", type: "number", min: 1, step: 1 },
  { key: "patience", path: "training.patience", label: "早停耐心值", type: "number", min: 0, step: 1 },
  { key: "save", path: "training.save", label: "保存模型", type: "enum", options: boolOptions },
  { key: "save_period", path: "training.save_period", label: "保存周期", type: "number", min: -1, step: 1 },
  { key: "cache", path: "training.cache", label: "缓存方式", type: "enum", options: cacheOptions },
  { key: "exist_ok", path: "training.exist_ok", label: "覆盖同名", type: "enum", options: boolOptions },
  { key: "pretrained", path: "training.pretrained", label: "预训练", type: "enum", options: boolOptions },
  { key: "optimizer", path: "training.optimizer", label: "优化器", type: "enum", options: optimizerOptions },
  { key: "verbose", path: "training.verbose", label: "详细日志", type: "enum", options: boolOptions },
  { key: "seed", path: "training.seed", label: "随机种子", type: "number", min: 0, step: 1 },
  { key: "deterministic", path: "training.deterministic", label: "确定性训练", type: "enum", options: boolOptions },
  { key: "single_cls", path: "training.single_cls", label: "单类别", type: "enum", options: boolOptions },
  { key: "rect", path: "training.rect", label: "矩形训练", type: "enum", options: boolOptions },
  { key: "cos_lr", path: "training.cos_lr", label: "余弦学习率", type: "enum", options: boolOptions },
  { key: "fraction", path: "training.fraction", label: "数据使用比例", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "profile", path: "training.profile", label: "性能分析", type: "enum", options: boolOptions },
  { key: "multi_scale", path: "training.multi_scale", label: "多尺度", type: "enum", options: boolOptions },
  { key: "compile", path: "training.compile", label: "模型编译", type: "enum", options: boolOptions },
  { key: "val", path: "training.val", label: "训练中验证", type: "enum", options: boolOptions },
  { key: "save_json", path: "training.save_json", label: "保存 JSON", type: "enum", options: boolOptions },
  { key: "conf", path: "training.conf", label: "置信度阈值", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "iou", path: "training.iou", label: "IoU 阈值", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "half", path: "training.half", label: "半精度", type: "enum", options: boolOptions },
  { key: "dnn", path: "training.dnn", label: "OpenCV DNN", type: "enum", options: boolOptions },
  { key: "plots", path: "training.plots", label: "生成图表", type: "enum", options: boolOptions },
  { key: "end2end", path: "training.end2end", label: "端到端", type: "enum", options: boolOptions }
];

const augmentationParams = [
  { key: "mosaic", path: "training.mosaic", label: "Mosaic", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "mixup", path: "training.mixup", label: "Mixup", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "close_mosaic", path: "training.close_mosaic", label: "关闭 Mosaic 轮数", type: "number", min: 0, step: 1 },
  { key: "hsv_h", path: "training.hsv_h", label: "HSV 色调", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "hsv_s", path: "training.hsv_s", label: "HSV 饱和度", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "hsv_v", path: "training.hsv_v", label: "HSV 明度", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "degrees", path: "training.degrees", label: "旋转角度", type: "number", min: 0, max: 180, step: 1 },
  { key: "translate", path: "training.translate", label: "平移比例", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "scale", path: "training.scale", label: "缩放比例", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "shear", path: "training.shear", label: "剪切角度", type: "number", min: 0, max: 180, step: 1 },
  { key: "perspective", path: "training.perspective", label: "透视", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "flipud", path: "training.flipud", label: "上下翻转", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "fliplr", path: "training.fliplr", label: "左右翻转", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "bgr", path: "training.bgr", label: "BGR 交换", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "cutmix", path: "training.cutmix", label: "CutMix", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "copy_paste", path: "training.copy_paste", label: "Copy-Paste", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "copy_paste_mode", path: "training.copy_paste_mode", label: "Copy-Paste 模式", type: "text" },
  { key: "auto_augment", path: "training.auto_augment", label: "自动增强", type: "text" },
  { key: "erasing", path: "training.erasing", label: "随机擦除", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "overlap_mask", path: "training.overlap_mask", label: "重叠掩码", type: "enum", options: boolOptions },
  { key: "mask_ratio", path: "training.mask_ratio", label: "掩码比例", type: "number", min: 0, step: 1 },
  { key: "dropout", path: "training.dropout", label: "Dropout", type: "number", min: 0, max: 1, step: 0.01 }
];

const optimizationParams = [
  { key: "momentum", path: "training.momentum", label: "动量 momentum", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "weight_decay", path: "training.weight_decay", label: "权重衰减", type: "number", min: 0, max: 0.1, step: 0.0001 },
  { key: "warmup_epochs", path: "training.warmup_epochs", label: "预热轮数", type: "number", min: 0, step: 1 },
  { key: "warmup_momentum", path: "training.warmup_momentum", label: "预热动量", type: "number", min: 0, max: 1, step: 0.01 },
  { key: "warmup_bias_lr", path: "training.warmup_bias_lr", label: "预热偏置学习率", type: "number", min: 0, step: 0.01 },
  { key: "box", path: "training.box", label: "Box 权重", type: "number", min: 0, step: 0.1 },
  { key: "cls", path: "training.cls", label: "Cls 权重", type: "number", min: 0, step: 0.1 },
  { key: "dfl", path: "training.dfl", label: "DFL 权重", type: "number", min: 0, step: 0.1 },
  { key: "pose", path: "training.pose", label: "Pose 权重", type: "number", min: 0, step: 0.1 },
  { key: "kobj", path: "training.kobj", label: "kobj 权重", type: "number", min: 0, step: 0.1 },
  { key: "rle", path: "training.rle", label: "RLE 系数", type: "number", min: 0, step: 0.1 },
  { key: "angle", path: "training.angle", label: "Angle 系数", type: "number", min: 0, step: 0.1 },
  { key: "nbs", path: "training.nbs", label: "标准批次 nbs", type: "number", min: 1, step: 1 }
];

const outputParams = [
  { key: "project", path: "output.project", label: "输出目录", type: "path", required: true, browseKind: "directory", filters: directoryFilters, placeholder: "runs", hint: "训练结果保存目录" },
  { key: "name", path: "output.name", label: "运行名称", type: "text", required: true, placeholder: "exp1", hint: "本次运行标识" }
];

const exportParams = [
  { key: "export_path", path: "export_model.export_path", label: "导出路径", type: "path", browseKind: "directory", filters: directoryFilters, placeholder: "runs/exports/exp1.onnx", hint: "导出目标路径" },
  { key: "format", path: "export_model.format", label: "导出格式", type: "enum", options: exportFormatOptions },
  { key: "keras", path: "export_model.keras", label: "导出 Keras", type: "enum", options: boolOptions },
  { key: "optimize", path: "export_model.optimize", label: "导出优化", type: "enum", options: boolOptions },
  { key: "int8", path: "export_model.int8", label: "INT8 量化", type: "enum", options: boolOptions },
  { key: "dynamic", path: "export_model.dynamic", label: "动态输入", type: "enum", options: boolOptions },
  { key: "simplify", path: "export_model.simplify", label: "简化模型", type: "enum", options: boolOptions },
  { key: "opset", path: "export_model.opset", label: "ONNX opset", type: "number", min: 0, step: 1 },
  { key: "workspace", path: "export_model.workspace", label: "工作空间", type: "number", min: 0, step: 1 },
  { key: "nms", path: "export_model.nms", label: "导出 NMS", type: "enum", options: boolOptions }
];

const displayParams = [
  { key: "source", path: "advanced.training.source", label: "推理数据 source", type: "path", browseKind: "file", filters: imageVideoFilters, placeholder: "datasets/mini_test/images/val/img_0000.jpg", hint: "推理输入路径或文件" },
  { key: "vid_stride", path: "advanced.training.vid_stride", label: "视频步长 vid_stride", type: "number", min: 1, step: 1 },
  { key: "stream_buffer", path: "advanced.training.stream_buffer", label: "流缓冲 stream_buffer", type: "enum", options: boolOptions },
  { key: "visualize", path: "advanced.training.visualize", label: "可视化 visualize", type: "enum", options: boolOptions },
  { key: "augment", path: "advanced.training.augment", label: "推理增强 augment", type: "enum", options: boolOptions },
  { key: "agnostic_nms", path: "advanced.training.agnostic_nms", label: "无类别 NMS", type: "enum", options: boolOptions },
  { key: "classes", path: "advanced.training.classes", label: "指定类别 classes", type: "text" },
  { key: "retina_masks", path: "advanced.training.retina_masks", label: "高分辨掩码 retina_masks", type: "enum", options: boolOptions },
  { key: "embed", path: "advanced.training.embed", label: "特征导出 embed", type: "text" },
  { key: "show", path: "advanced.training.show", label: "实时预览 show", type: "enum", options: boolOptions },
  { key: "save_frames", path: "advanced.training.save_frames", label: "保存帧 save_frames", type: "enum", options: boolOptions },
  { key: "save_txt", path: "advanced.training.save_txt", label: "保存标签 save_txt", type: "enum", options: boolOptions },
  { key: "save_conf", path: "advanced.training.save_conf", label: "保存置信度 save_conf", type: "enum", options: boolOptions },
  { key: "save_crop", path: "advanced.training.save_crop", label: "保存裁剪 save_crop", type: "enum", options: boolOptions },
  { key: "show_labels", path: "advanced.training.show_labels", label: "显示标签 show_labels", type: "enum", options: boolOptions },
  { key: "show_conf", path: "advanced.training.show_conf", label: "显示置信度 show_conf", type: "enum", options: boolOptions },
  { key: "show_boxes", path: "advanced.training.show_boxes", label: "显示检测框 show_boxes", type: "enum", options: boolOptions },
  { key: "line_width", path: "advanced.training.line_width", label: "框线宽 line_width", type: "number", min: 0, step: 1 }
];

const cloudParams = [
  { key: "compute_provider", path: "training.compute_provider", label: "计算环境", type: "enum", options: computeProviderOptions },
  { key: "base_url", path: "training.cloud_api.base_url", label: "云训练 Base URL", type: "text" },
  { key: "submit_path", path: "training.cloud_api.submit_path", label: "提交路径", type: "text" },
  { key: "status_path", path: "training.cloud_api.status_path", label: "状态路径", type: "text" },
  { key: "logs_path", path: "training.cloud_api.logs_path", label: "日志路径", type: "text" },
  { key: "stop_path", path: "training.cloud_api.stop_path", label: "停止路径", type: "text" },
  { key: "token", path: "training.cloud_api.token", label: "云训练 Token", type: "text" },
  { key: "poll_interval_sec", path: "training.cloud_api.poll_interval_sec", label: "轮询间隔(秒)", type: "number", min: 1, step: 1 },
  { key: "enabled", path: "training.dataset_api.enabled", label: "启用数据集 API", type: "enum", options: boolOptions },
  { key: "source", path: "training.dataset_api.source", label: "数据集来源", type: "enum", options: [
    { value: "path", label: "本地路径" },
    { value: "api", label: "API" }
  ] },
  { key: "resolve_url", path: "training.dataset_api.resolve_url", label: "数据集解析 URL", type: "text" },
  { key: "token", path: "training.dataset_api.token", label: "数据集 API Token", type: "text" },
  { key: "dataset_name", path: "training.dataset_api.dataset_name", label: "数据集名称", type: "text" }
];

const wandbParams = [
  { key: "enabled", path: "wandb.enabled", label: "启用 W&B", type: "enum", options: boolOptions },
  { key: "mode", path: "wandb.mode", label: "W&B 模式", type: "enum", options: wandbModeOptions },
  { key: "project", path: "wandb.project", label: "项目名", type: "text" },
  { key: "entity", path: "wandb.entity", label: "Entity", type: "text" },
  { key: "name", path: "wandb.name", label: "运行名", type: "text" },
  { key: "group", path: "wandb.group", label: "分组", type: "text" },
  { key: "job_type", path: "wandb.job_type", label: "Job 类型", type: "text" },
  { key: "tags", path: "wandb.tags", label: "Tags", type: "text" },
  { key: "notes", path: "wandb.notes", label: "备注", type: "text" }
];

export const BASIC_DISTILLATION_KEYS = new Set(distillationParams.map((param) => param.key).filter((key) => !["teacher_cfg", "teacher_device", "feature_layers", "temperature", "alpha", "beta", "gamma"].includes(key)));

export const BASIC_TRAINING_KEYS = new Set([
  "compute_provider",
  "cloud_api",
  "dataset_api",
  "data_yaml",
  "device",
  "epochs",
  "imgsz",
  "batch",
  "max_det",
  "lr0",
  "lrf",
  "warmup_epochs",
  "mosaic",
  "mixup",
  "close_mosaic",
  "amp"
]);

export const TRAINING_ADVANCED_SECTIONS = [
  { title: "训练核心", params: trainingCoreParams },
  { title: "数据增强", params: augmentationParams },
  { title: "优化与损失", params: optimizationParams }
];

export const DISPLAY_ADVANCED_SECTIONS = [
  { title: "推理与展示", params: displayParams }
];

export const EXPORT_ADVANCED_SECTIONS = [
  { title: "导出参数", params: exportParams }
];

export const DISTILLATION_ADVANCED_SECTIONS = [
  { title: "蒸馏配置", params: distillationParams }
];

export const CONFIG_GROUPS = [
  {
    id: "distillation",
    title: "蒸馏配置",
    icon: "school",
    description: "教师-学生模型与蒸馏损失参数",
    priority: "high",
    params: distillationParams
  },
  {
    id: "training",
    title: "基础训练",
    icon: "fitness_center",
    description: "训练轮数、设备、数据集、保存策略",
    priority: "high",
    params: trainingCoreParams
  },
  {
    id: "augmentation",
    title: "数据增强",
    icon: "auto_fix_high",
    description: "Mosaic、Mixup、HSV 与几何增强",
    priority: "medium",
    params: augmentationParams
  },
  {
    id: "optimization",
    title: "损失与优化",
    icon: "tune",
    description: "学习率、动量、损失权重",
    priority: "medium",
    params: optimizationParams
  },
  {
    id: "output",
    title: "输出配置",
    icon: "output",
    description: "输出目录与运行名称",
    priority: "high",
    params: outputParams
  },
  {
    id: "export",
    title: "模型导出",
    icon: "file_upload",
    description: "导出格式、量化与优化",
    priority: "low",
    params: exportParams
  },
  {
    id: "display",
    title: "推理与展示",
    icon: "visibility",
    description: "推理 source 与展示参数",
    priority: "low",
    params: displayParams
  },
  {
    id: "cloud",
    title: "云端与数据 API",
    icon: "cloud",
    description: "远程训练与数据集解析配置",
    priority: "low",
    params: cloudParams
  },
  {
    id: "wandb",
    title: "实验追踪",
    icon: "insights",
    description: "Weights & Biases 记录配置",
    priority: "low",
    params: wandbParams
  }
];

export const getAllParams = () => {
  const all = [];
  CONFIG_GROUPS.forEach((group) => {
    group.params.forEach((param) => {
      all.push({
        ...param,
        groupId: group.id,
        groupTitle: group.title,
      });
    });
  });
  return all;
};

export const getParamByKey = (key) => getAllParams().find((param) => param.key === key);
