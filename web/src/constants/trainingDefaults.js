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
    max_det: 300,
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
  export_model: {
    export_path: "",
    format: "onnx",
    keras: false,
    optimize: false,
    int8: false,
    dynamic: false,
    simplify: false,
    opset: 0,
    workspace: 0,
    nms: false
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
