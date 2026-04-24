export function QuickStat({ label, value }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value text-primary">{value}</span>
    </div>
  );
}
