export function detectLogLevel(line) {
  const text = String(line || "");
  if (/\b(error|exception|traceback|failed?)\b/i.test(text)) return "error";
  if (/(\bwarn(ing)?\b|caution|警告|告警|⚠|\[W\]|^\s*W\d*:|\bignoring\b|忽略|已忽略|\bdeprecated\b)/i.test(text)) {
    return "warning";
  }
  if (/\b(success|done|completed?)\b/i.test(text)) return "success";
  return "info";
}
