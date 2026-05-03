export default function ConfigNav({ groups, activeGroup, onNavigate }) {
  return (
    <aside className="config-nav-panel config-card">
      <div className="card-header">配置分组</div>
      <div className="config-nav-list">
        {groups.map((group) => {
          const isActive = activeGroup === group.id;
          return (
            <button
              key={group.id}
              type="button"
              className={`config-nav-item ${isActive ? "active" : ""}`}
              onClick={() => onNavigate(group.id)}
            >
              <span className="material-icons">{group.icon || "tune"}</span>
              <span className="config-nav-item-text">
                <strong>{group.title}</strong>
                <small>{group.description}</small>
              </span>
              {group.priority === "high" ? <span className="config-nav-badge">必填</span> : null}
            </button>
          );
        })}
      </div>
    </aside>
  );
}
