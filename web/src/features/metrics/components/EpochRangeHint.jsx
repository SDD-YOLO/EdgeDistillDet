function normalizeEpochRange(range) {
  if (range == null || range === "") return "all";
  const s = String(range).trim();
  if (s === "last30") return "last30";
  if (s === "last10") return "last10";
  return "all";
}

function visibleEpochCount(total, range) {
  const r = normalizeEpochRange(range);
  if (!total) return 0;
  if (r === "last30") return Math.min(30, total);
  if (r === "last10") return Math.min(10, total);
  return total;
}

export function EpochRangeHint({ total, range, unit = "epoch" }) {
  const r = normalizeEpochRange(range);
  if (!total || r === "all") return null;
  const vis = visibleEpochCount(total, range);
  const sameAsAll =
    (r === "last30" && total <= 30) || (r === "last10" && total <= 10);
  return (
    <span className="chart-epoch-hint">
      显示 {vis}/{total} 个{unit}
      {sameAsAll ? "（与「全部」相同）" : ""}
    </span>
  );
}
