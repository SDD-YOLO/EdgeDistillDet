export function OverviewCard({ icon, label, value }) {
  return (
    <div className="overview-card">
      <div className="overview-icon"><span className="material-icons">{icon}</span></div>
      <div className="overview-info">
        <span className="overview-label">{label}</span>
        <span className="overview-value">{value}</span>
      </div>
    </div>
  );
}
