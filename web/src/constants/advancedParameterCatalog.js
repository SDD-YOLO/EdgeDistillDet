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

export const BASIC_DISTILLATION_KEYS = new Set([
  "student_weight",
  "teacher_weight",
  "alpha_init",
  "T_max",
  "T_min",
  "warm_epochs",
  "w_kd",
  "w_focal",
  "w_feat",
  "scale_boost",
  "focal_gamma"
]);

const boolOptions = [
  { value: "true", label: "true" },
  { value: "false", label: "false" }
];

const exportFormatOptions = [
  { value: "onnx", label: "ONNX" },
  { value: "torchscript", label: "TorchScript" }
];

export const TRAINING_ADVANCED_SECTIONS = [
  {
    title: "训练运行参数",
    params: [
      { key: "task", label: "任务类型 task", type: "enum", options: [{ value: "detect", label: "detect" }, { value: "segment", label: "segment" }] },
      { key: "mode", label: "运行模式 mode", type: "enum", options: [{ value: "train", label: "train" }, { value: "val", label: "val" }, { value: "predict", label: "predict" }] },
      { key: "time", label: "限时训练 time", type: "number" },
      { key: "patience", label: "早停 patience", type: "number" },
      { key: "save", label: "保存模型 save", type: "enum", options: boolOptions },
      { key: "save_period", label: "阶段保存 save_period", type: "number" },
      { key: "cache", label: "数据缓存 cache", type: "enum", options: [{ value: "false", label: "false" }, { value: "ram", label: "ram" }, { value: "disk", label: "disk" }] },
      { key: "workers", label: "数据线程 workers", type: "number" },
      { key: "exist_ok", label: "覆盖同名 exist_ok", type: "enum", options: boolOptions },
      { key: "pretrained", label: "预训练 pretrained", type: "enum", options: boolOptions },
      { key: "optimizer", label: "优化器 optimizer", type: "enum", options: [{ value: "auto", label: "auto" }, { value: "SGD", label: "SGD" }, { value: "Adam", label: "Adam" }] },
      { key: "verbose", label: "详细日志 verbose", type: "enum", options: boolOptions },
      { key: "seed", label: "随机种子 seed", type: "number" },
      { key: "deterministic", label: "固定随机 deterministic", type: "enum", options: boolOptions },
      { key: "single_cls", label: "单类别 single_cls", type: "enum", options: boolOptions },
      { key: "rect", label: "矩形训练 rect", type: "enum", options: boolOptions },
      { key: "cos_lr", label: "余弦学习率 cos_lr", type: "enum", options: boolOptions },
      { key: "fraction", label: "数据使用比例 fraction", type: "number" },
      { key: "profile", label: "性能分析 profile", type: "enum", options: boolOptions },
      { key: "freeze", label: "冻结层 freeze", type: "text" },
      { key: "multi_scale", label: "多尺度 multi_scale", type: "enum", options: boolOptions },
      { key: "compile", label: "模型编译 compile", type: "enum", options: boolOptions }
    ]
  },
  {
    title: "数据增强与追踪参数",
    params: [
      { key: "hsv_h", label: "hsv_h", type: "number" },
      { key: "hsv_s", label: "hsv_s", type: "number" },
      { key: "hsv_v", label: "hsv_v", type: "number" },
      { key: "degrees", label: "degrees", type: "number" },
      { key: "translate", label: "translate", type: "number" },
      { key: "scale", label: "scale", type: "number" },
      { key: "shear", label: "shear", type: "number" },
      { key: "perspective", label: "perspective", type: "number" },
      { key: "flipud", label: "flipud", type: "number" },
      { key: "fliplr", label: "fliplr", type: "number" },
      { key: "bgr", label: "bgr", type: "number" },
      { key: "cutmix", label: "cutmix", type: "number" },
      { key: "copy_paste", label: "copy_paste", type: "number" },
      { key: "copy_paste_mode", label: "copy_paste_mode", type: "text" },
      { key: "auto_augment", label: "auto_augment", type: "text" },
      { key: "erasing", label: "erasing", type: "number" },
      { key: "cfg", label: "模型配置 cfg", type: "text" },
      { key: "tracker", label: "追踪器 tracker", type: "text" },
      { key: "save_dir", label: "保存路径 save_dir", type: "text" },
      { key: "overlap_mask", label: "overlap_mask", type: "enum", options: boolOptions },
      { key: "mask_ratio", label: "mask_ratio", type: "number" },
      { key: "dropout", label: "dropout", type: "number" }
    ]
  },
  {
    title: "训练损失与优化参数",
    params: [
      { key: "momentum", label: "动量 momentum", type: "number" },
      { key: "weight_decay", label: "权重衰减 weight_decay", type: "number" },
      { key: "warmup_momentum", label: "预热动量 warmup_momentum", type: "number" },
      { key: "warmup_bias_lr", label: "偏置预热 warmup_bias_lr", type: "number" },
      { key: "box", label: "box 损失权重", type: "number" },
      { key: "cls", label: "cls 损失权重", type: "number" },
      { key: "dfl", label: "dfl 权重", type: "number" },
      { key: "pose", label: "pose 权重", type: "number" },
      { key: "kobj", label: "kobj 权重", type: "number" },
      { key: "rle", label: "rle 系数", type: "number" },
      { key: "angle", label: "angle 系数", type: "number" },
      { key: "nbs", label: "标准批次 nbs", type: "number" }
    ]
  }
];

export const EXPORT_ADVANCED_SECTIONS = [
  {
    title: "导出参数",
    params: [
      { key: "export_path", label: "导出路径 export_path", type: "path" },
      { key: "format", label: "导出格式 format", type: "enum", options: exportFormatOptions },
      { key: "keras", label: "导出 keras", type: "enum", options: boolOptions },
      { key: "optimize", label: "导出优化 optimize", type: "enum", options: boolOptions },
      { key: "int8", label: "INT8 量化 int8", type: "enum", options: boolOptions },
      { key: "dynamic", label: "动态输入 dynamic", type: "enum", options: boolOptions },
      { key: "simplify", label: "简化模型 simplify", type: "enum", options: boolOptions },
      { key: "opset", label: "ONNX opset", type: "number" },
      { key: "workspace", label: "工作空间 workspace", type: "number" },
      { key: "nms", label: "导出 NMS nms", type: "enum", options: boolOptions }
    ]
  }
];

export const DISPLAY_ADVANCED_SECTIONS = [
  {
    title: "推理与结果展示参数",
    params: [
      { key: "source", label: "推理数据 source", type: "path" },
      { key: "vid_stride", label: "视频步长 vid_stride", type: "number" },
      { key: "stream_buffer", label: "流缓冲 stream_buffer", type: "enum", options: boolOptions },
      { key: "visualize", label: "特征可视化 visualize", type: "enum", options: boolOptions },
      { key: "augment", label: "推理增强 augment", type: "enum", options: boolOptions },
      { key: "agnostic_nms", label: "无类别 NMS agnostic_nms", type: "enum", options: boolOptions },
      { key: "classes", label: "指定类别 classes", type: "text" },
      { key: "retina_masks", label: "高分辨掩码 retina_masks", type: "enum", options: boolOptions },
      { key: "embed", label: "特征导出 embed", type: "text" },
      { key: "show", label: "实时预览 show", type: "enum", options: boolOptions },
      { key: "save_frames", label: "保存帧 save_frames", type: "enum", options: boolOptions },
      { key: "save_txt", label: "保存标签 save_txt", type: "enum", options: boolOptions },
      { key: "save_conf", label: "保存置信度 save_conf", type: "enum", options: boolOptions },
      { key: "save_crop", label: "保存裁剪 save_crop", type: "enum", options: boolOptions },
      { key: "show_labels", label: "显示标签 show_labels", type: "enum", options: boolOptions },
      { key: "show_conf", label: "显示置信度 show_conf", type: "enum", options: boolOptions },
      { key: "show_boxes", label: "显示检测框 show_boxes", type: "enum", options: boolOptions },
      { key: "line_width", label: "框线宽 line_width", type: "number" }
    ]
  }
];

export const DISTILLATION_ADVANCED_SECTIONS = [
  {
    title: "蒸馏扩展参数",
    params: [
      { key: "teacher_cfg", label: "teacher_cfg", type: "text" },
      { key: "distill_mode", label: "distill_mode", type: "enum", options: [{ value: "adaptive", label: "adaptive" }, { value: "feature", label: "feature" }, { value: "response", label: "response" }] },
      { key: "temperature", label: "temperature", type: "number" },
      { key: "alpha", label: "alpha", type: "number" },
      { key: "beta", label: "beta", type: "number" },
      { key: "gamma", label: "gamma", type: "number" },
      { key: "cls_distill_weight", label: "cls_distill_weight", type: "number" },
      { key: "box_distill_weight", label: "box_distill_weight", type: "number" },
      { key: "obj_distill_weight", label: "obj_distill_weight", type: "number" },
      { key: "distill_loss_type", label: "distill_loss_type", type: "enum", options: [{ value: "kl", label: "kl" }, { value: "mse", label: "mse" }, { value: "l1", label: "l1" }] },
      { key: "freeze_teacher", label: "freeze_teacher", type: "enum", options: boolOptions },
      { key: "teacher_device", label: "teacher_device", type: "text" },
      { key: "distill_start_epoch", label: "distill_start_epoch", type: "number" },
      { key: "distill_end_epoch", label: "distill_end_epoch", type: "number" },
      { key: "dynamic_alpha", label: "dynamic_alpha", type: "enum", options: boolOptions },
      { key: "dynamic_temperature", label: "dynamic_temperature", type: "enum", options: boolOptions },
      { key: "feature_layers", label: "feature_layers", type: "text" },
      { key: "lambda_kd", label: "lambda_kd", type: "number" },
      { key: "use_adaptive_loss", label: "use_adaptive_loss", type: "enum", options: boolOptions }
    ]
  }
];
