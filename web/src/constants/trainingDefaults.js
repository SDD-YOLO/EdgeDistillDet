export const DEFAULT_FORM = {
  distillation: {
    student_weight: "",
    teacher_weight: "",
    alpha_init: 0.5,
    T_max: 6,
    T_min: 1.5,
    warm_epochs: 5,
    w_kd: 0.5,
    w_focal: 0.3,
    w_feat: 0,
    scale_boost: 2,
    focal_gamma: 2
  },
  training: {
    compute_provider: "local",
    cloud_api: {
      base_url: "",
      submit_path: "/train/start",
      status_path: "/train/status",
      logs_path: "/train/logs",
      stop_path: "/train/stop",
      token: "",
      poll_interval_sec: 3
    },
    dataset_api: {
      enabled: false,
      source: "path",
      resolve_url: "",
      token: "",
      dataset_name: ""
    },
    data_yaml: "",
    device: "0",
    epochs: 150,
    imgsz: 640,
    batch: -1,
    lr0: 0.01,
    lrf: 0.1,
    warmup_epochs: 3,
    mosaic: 0.8,
    mixup: 0.1,
    close_mosaic: 20,
    amp: true
  },
  output: {
    project: "runs",
    name: "exp1"
  },
  wandb: {
    enabled: false,
    mode: "online",
    project: "edge-distilldet",
    entity: "",
    name: "",
    group: "",
    job_type: "distill-train",
    tags: "",
    notes: ""
  },
  advanced: {
    training: {
      task: "detect",
      mode: "train",
      time: 0,
      patience: 0,
      save: false,
      save_period: 0,
      cache: false,
      workers: 0,
      exist_ok: false,
      pretrained: false,
      optimizer: "auto",
      verbose: false,
      seed: 0,
      deterministic: false,
      single_cls: false,
      rect: false,
      cos_lr: false,
      fraction: 0,
      profile: false,
      freeze: "",
      multi_scale: false,
      compile: false,
      val: false,
      split: "",
      save_json: false,
      conf: 0,
      iou: 0,
      max_det: 0,
      half: false,
      dnn: false,
      plots: false,
      end2end: false,
      source: "",
      vid_stride: 0,
      stream_buffer: false,
      visualize: false,
      augment: false,
      agnostic_nms: false,
      classes: "",
      retina_masks: false,
      embed: "",
      show: false,
      save_frames: false,
      save_txt: false,
      save_conf: false,
      save_crop: false,
      show_labels: false,
      show_conf: false,
      show_boxes: false,
      line_width: 0,
      export_path: "",
      format: "onnx",
      keras: false,
      optimize: false,
      int8: false,
      dynamic: false,
      simplify: false,
      opset: 0,
      workspace: 0,
      nms: false,
      momentum: 0,
      weight_decay: 0,
      warmup_momentum: 0,
      warmup_bias_lr: 0,
      box: 0,
      cls: 0,
      dfl: 0,
      pose: 0,
      kobj: 0,
      rle: 0,
      angle: 0,
      nbs: 0,
      hsv_h: 0,
      hsv_s: 0,
      hsv_v: 0,
      degrees: 0,
      translate: 0,
      scale: 0,
      shear: 0,
      perspective: 0,
      flipud: 0,
      fliplr: 0,
      bgr: 0,
      cutmix: 0,
      copy_paste: 0,
      copy_paste_mode: "",
      auto_augment: "",
      erasing: 0,
      cfg: "",
      tracker: "",
      save_dir: "",
      overlap_mask: false,
      mask_ratio: 0,
      dropout: 0
    },
    distillation: {
      teacher_cfg: "",
      distill_mode: "adaptive",
      temperature: 0,
      alpha: 0,
      beta: 0,
      gamma: 0,
      cls_distill_weight: 0,
      box_distill_weight: 0,
      obj_distill_weight: 0,
      distill_loss_type: "kl",
      freeze_teacher: false,
      teacher_device: "",
      distill_start_epoch: 0,
      distill_end_epoch: 0,
      dynamic_alpha: false,
      dynamic_temperature: false,
      feature_layers: "",
      lambda_kd: 0,
      use_adaptive_loss: false
    }
  }
};

export const COMPUTE_PRESETS = {
  local: {
    device: "0",
    outputProject: "runs"
  },
  autodl: {
    device: "0",
    outputProject: "/root/autodl-tmp/runs"
  },
  colab: {
    device: "0",
    outputProject: "/content/runs"
  }
};

export function inferComputeProviderFromConfig(config, fallback = "local") {
  const explicitProvider = String(config?.training?.compute_provider || "").trim().toLowerCase();
  if (explicitProvider === "autodl" || explicitProvider === "colab" || explicitProvider === "local" || explicitProvider === "remote_api") {
    return explicitProvider;
  }

  const outputProject = String(config?.output?.project || "").toLowerCase();
  const dataYaml = String(config?.training?.data_yaml || "").toLowerCase();
  const featureText = `${outputProject} ${dataYaml}`;

  if (featureText.includes("/root/autodl-tmp") || featureText.includes("autodl")) {
    return "autodl";
  }
  if (featureText.includes("/content/") || featureText.includes("colab")) {
    return "colab";
  }
  if (featureText.includes("http://") || featureText.includes("https://")) {
    return "remote_api";
  }
  return fallback;
}
