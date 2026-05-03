import { DEFAULT_FORM, inferComputeProviderFromConfig } from "../../../constants/trainingDefaults";
import {
  BASIC_DISTILLATION_KEYS,
  BASIC_TRAINING_KEYS,
  DISTILLATION_ADVANCED_SECTIONS,
  DISPLAY_ADVANCED_SECTIONS,
  EXPORT_ADVANCED_SECTIONS,
  TRAINING_ADVANCED_SECTIONS
} from "../../../constants/configGroups";

export function normalizeWandbForUi(wandb) {
  if (!wandb || typeof wandb !== "object") return wandb;
  const next = { ...wandb };
  if (Array.isArray(next.tags)) {
    next.tags = next.tags.map((t) => String(t).trim()).filter(Boolean).join(", ");
  }
  return next;
}

export function omitKeys(source, keys) {
  const out = { ...(source || {}) };
  keys.forEach((key) => delete out[key]);
  return out;
}

export function normalizeAdvancedValueForUi(value) {
  if (value === undefined || value === null) return "";
  if (typeof value === "boolean" || typeof value === "number") return value;
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function extractAdvancedValues(sectionConfig, basicKeys, sections) {
  const config = sectionConfig && typeof sectionConfig === "object" ? sectionConfig : {};
  const supportedKeys = new Set();
  sections.forEach((section) => {
    (section?.params || []).forEach((param) => supportedKeys.add(param.key));
  });
  const out = {};
  Object.entries(config).forEach(([key, value]) => {
    if (basicKeys.has(key)) return;
    if (!supportedKeys.has(key)) return;
    out[key] = normalizeAdvancedValueForUi(value);
  });
  return out;
}

export function parseAdvancedValue(raw) {
  if (raw === null || raw === undefined) return undefined;
  if (typeof raw === "boolean" || typeof raw === "number") return raw;
  const text = String(raw).trim();
  if (!text) return undefined;
  const lower = text.toLowerCase();
  if (lower === "true") return true;
  if (lower === "false") return false;
  if (lower === "null" || lower === "none") return null;
  if (/^-?\d+(\.\d+)?$/.test(text)) return Number(text);
  if ((text.startsWith("{") && text.endsWith("}")) || (text.startsWith("[") && text.endsWith("]")) || (text.startsWith("\"") && text.endsWith("\""))) {
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }
  return text;
}

export function applyAdvancedOverrides(target, advanced) {
  const merged = { ...(target || {}) };
  Object.entries(advanced || {}).forEach(([key, rawValue]) => {
    const parsed = parseAdvancedValue(rawValue);
    if (parsed === undefined) {
      delete merged[key];
      return;
    }
    merged[key] = parsed;
  });
  return merged;
}

export function mergeDistillConfigIntoForm(prev, config) {
  if (!config || typeof config !== "object") return prev;
  const inferredProvider = inferComputeProviderFromConfig(config, prev.training?.compute_provider || "local");
  const incomingDistillation = omitKeys(config?.distillation, ["temperature", "schedule_type", "feat_layer", "alpha", "distill_type"]);
  const incomingTraining = omitKeys(config?.training, ["warm_epochs", "workers", "batch_size", "learning_rate", "grad_clip"]);
  const incomingOutput = omitKeys(config?.output, ["model_dir", "log_dir"]);
  const incomingAdvancedTraining = {
    ...extractAdvancedValues(incomingTraining, BASIC_TRAINING_KEYS, TRAINING_ADVANCED_SECTIONS),
    ...extractAdvancedValues(incomingTraining, BASIC_TRAINING_KEYS, DISPLAY_ADVANCED_SECTIONS),
    ...extractAdvancedValues(incomingTraining, BASIC_TRAINING_KEYS, EXPORT_ADVANCED_SECTIONS),
    ...extractAdvancedValues(config?.advanced?.training, BASIC_TRAINING_KEYS, TRAINING_ADVANCED_SECTIONS),
    ...extractAdvancedValues(config?.advanced?.training, BASIC_TRAINING_KEYS, DISPLAY_ADVANCED_SECTIONS),
    ...extractAdvancedValues(config?.advanced?.training, BASIC_TRAINING_KEYS, EXPORT_ADVANCED_SECTIONS)
  };
  const incomingExportModel = {
    ...extractAdvancedValues(config?.export_model, BASIC_TRAINING_KEYS, EXPORT_ADVANCED_SECTIONS),
    ...extractAdvancedValues(config?.advanced?.training, BASIC_TRAINING_KEYS, EXPORT_ADVANCED_SECTIONS)
  };
  const incomingAdvancedDistillation = {
    ...extractAdvancedValues(incomingDistillation, BASIC_DISTILLATION_KEYS, DISTILLATION_ADVANCED_SECTIONS),
    ...extractAdvancedValues(config?.advanced?.distillation, BASIC_DISTILLATION_KEYS, DISTILLATION_ADVANCED_SECTIONS)
  };
  if (
    Object.prototype.hasOwnProperty.call(incomingTraining, "warm_epochs") &&
    !Object.prototype.hasOwnProperty.call(incomingTraining, "warmup_epochs")
  ) {
    incomingTraining.warmup_epochs = incomingTraining.warm_epochs;
  }
  delete incomingTraining.warm_epochs;
  return {
    ...prev,
    distillation: { ...prev.distillation, ...incomingDistillation },
    training: {
      ...prev.training,
      ...incomingTraining,
      cloud_api: { ...prev.training?.cloud_api, ...(incomingTraining?.cloud_api || {}) },
      dataset_api: { ...prev.training?.dataset_api, ...(incomingTraining?.dataset_api || {}) },
      compute_provider: inferredProvider
    },
    output: { ...prev.output, ...incomingOutput },
    export_model: { ...(prev.export_model || {}), ...incomingExportModel },
    wandb: { ...prev.wandb, ...normalizeWandbForUi(config?.wandb) },
    advanced: {
      training: { ...(prev.advanced?.training || {}), ...incomingAdvancedTraining },
      distillation: { ...(prev.advanced?.distillation || {}), ...incomingAdvancedDistillation }
    }
  };
}

export function buildConfigPayload(sourceForm) {
  const cloned = JSON.parse(JSON.stringify(sourceForm || {}));
  const payload = {
    distillation: omitKeys(cloned.distillation, ["temperature", "schedule_type", "feat_layer", "alpha", "distill_type"]),
    training: omitKeys(cloned.training, ["warm_epochs", "workers", "batch_size", "learning_rate", "grad_clip"]),
    output: omitKeys(cloned.output, ["model_dir", "log_dir"]),
    wandb: cloned.wandb || {}
  };
  const wandb = payload.wandb || {};
  const rawTags = wandb.tags;
  if (typeof rawTags === "string") {
    wandb.tags = rawTags.split(",").map((s) => s.trim()).filter(Boolean);
  } else if (!Array.isArray(rawTags)) {
    wandb.tags = [];
  }
  payload.wandb = wandb;
  const exportAdvancedKeys = new Set(EXPORT_ADVANCED_SECTIONS.flatMap((section) => (section.params || []).map((param) => param.key)));
  const advancedTraining = { ...(cloned?.advanced?.training || {}) };
  const explicitExportModel = { ...(cloned?.export_model || {}) };
  Object.entries(advancedTraining).forEach(([key, value]) => {
    const explicitValue = explicitExportModel[key];
    if (explicitValue === undefined || explicitValue === "" || exportAdvancedKeys.has(key)) {
      if (value !== undefined) {
        explicitExportModel[key] = value;
      }
    }
  });
  exportAdvancedKeys.forEach((key) => delete advancedTraining[key]);

  payload.export_model = explicitExportModel;
  payload.training = applyAdvancedOverrides(payload.training, advancedTraining);
  payload.distillation = applyAdvancedOverrides(payload.distillation, cloned?.advanced?.distillation || {});
  payload.advanced = {
    training: advancedTraining,
    distillation: cloned?.advanced?.distillation || {}
  };
  return payload;
}
