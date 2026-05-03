import { useMemo } from "react";

function toYaml(value, indent = 0) {
  const spaces = "  ".repeat(indent);
  if (value === null || value === undefined || value === "") return "";
  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    return value
      .map(
        (item) =>
          `${spaces}- ${
            typeof item === "object"
              ? `\n${toYaml(item, indent + 1)}`
              : String(item)
          }`,
      )
      .join("\n");
  }
  if (typeof value !== "object") {
    return String(value);
  }
  return Object.entries(value)
    .filter(
      ([, nested]) => nested !== null && nested !== undefined && nested !== "",
    )
    .map(([key, nested]) => {
      if (typeof nested === "object" && !Array.isArray(nested)) {
        const nestedYaml = toYaml(nested, indent + 1);
        return `${spaces}${key}:\n${
          nestedYaml ? `${nestedYaml}\n` : ""
        }`.trimEnd();
      }
      if (Array.isArray(nested)) {
        const nestedYaml = toYaml(nested, indent + 1);
        return `${spaces}${key}:\n${nestedYaml}`;
      }
      return `${spaces}${key}: ${String(nested)}`;
    })
    .join("\n");
}

export default function YamlPreview({ payload, title = "实时 YAML 预览" }) {
  const yamlText = useMemo(() => toYaml(payload), [payload]);

  return (
    <aside className="yaml-preview-panel config-card">
      <div className="card-header">{title}</div>
      <pre className="yaml-preview-body">{yamlText || "# 暂无配置数据"}</pre>
    </aside>
  );
}
