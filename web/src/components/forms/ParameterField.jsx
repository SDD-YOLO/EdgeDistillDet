import { TextField } from "./TextField";
import { SelectField } from "./SelectField";
import { NumberField } from "./NumberField";
import { PathField } from "./PathField";

function normalizeValueForEnum(value) {
  if (value === undefined || value === null) return "";
  if (typeof value === "boolean") return String(value);
  return String(value);
}

export function ParameterField({
  param,
  value,
  onChange,
  onBrowse,
  disabled = false,
  title = "",
}) {
  const fieldType = param?.type || "text";
  const label = param?.label || param?.key || "参数";
  if (fieldType === "enum") {
    return (
      <SelectField
        label={label}
        value={normalizeValueForEnum(value)}
        onChange={(next) => onChange(next)}
        options={param.options || []}
        disabled={disabled}
        title={title}
      />
    );
  }

  if (fieldType === "number") {
    return (
      <NumberField
        label={label}
        value={value === "" ? null : value}
        step={param.step ?? "any"}
        onChange={(next) => onChange(next === null ? "" : next)}
        disabled={disabled}
        title={title}
      />
    );
  }

  if (fieldType === "path") {
    return (
      <PathField
        label={label}
        value={String(value || "")}
        onChange={(next) => onChange(next)}
        onBrowse={onBrowse}
        disabled={disabled}
      />
    );
  }

  return (
    <TextField
      label={label}
      value={String(value || "")}
      onChange={(next) => onChange(next)}
      disabled={disabled}
    />
  );
}
