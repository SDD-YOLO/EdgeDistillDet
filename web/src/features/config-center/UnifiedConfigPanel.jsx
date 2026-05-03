import { useEffect, useMemo } from "react";
import { ParameterField } from "../../components/forms/ParameterField";

export default function UnifiedConfigPanel({
  groups,
  form,
  getValue,
  setValue,
  pickLocalPath,
  running,
  activeGroup,
  searchQuery,
  expandedGroups,
  setExpandedGroups,
}) {
  const visibleGroups = useMemo(() => {
    if (!activeGroup) return groups;
    return groups.filter((group) => group.id === activeGroup);
  }, [groups, activeGroup]);

  const filteredGroups = useMemo(() => {
    if (!searchQuery?.trim()) return visibleGroups;
    const query = searchQuery.toLowerCase();
    return visibleGroups
      .map((group) => ({
        ...group,
        params: (group.params || []).filter((param) => {
          const label = String(param.label || "").toLowerCase();
          const key = String(param.key || param.path || "").toLowerCase();
          const hint = String(param.hint || "").toLowerCase();
          return (
            label.includes(query) || key.includes(query) || hint.includes(query)
          );
        }),
      }))
      .filter((group) => group.params.length > 0);
  }, [visibleGroups, searchQuery]);

  const highlightedKeys = useMemo(() => {
    if (!searchQuery?.trim()) return new Set();
    const query = searchQuery.toLowerCase();
    return new Set(
      visibleGroups
        .flatMap((group) => group.params || [])
        .filter((param) => {
          const label = String(param.label || "").toLowerCase();
          const key = String(param.key || param.path || "").toLowerCase();
          const hint = String(param.hint || "").toLowerCase();
          return (
            label.includes(query) || key.includes(query) || hint.includes(query)
          );
        })
        .map((param) => param.path || param.key),
    );
  }, [visibleGroups, searchQuery]);

  const toggleGroup = (groupId) => {
    setExpandedGroups((prev) =>
      prev.includes(groupId)
        ? prev.filter((id) => id !== groupId)
        : [...prev, groupId],
    );
  };

  useEffect(() => {
    if (!activeGroup) return;
    setExpandedGroups((prev) =>
      prev.includes(activeGroup) ? prev : [...prev, activeGroup],
    );
    const el = document.getElementById(`config-group-${activeGroup}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [activeGroup, setExpandedGroups]);

  const handleBrowse = async (param) => {
    const title = `选择${param.label}`;
    const selected = await pickLocalPath({
      kind: param.browseKind || "file",
      title,
      initialPath: String(getValue(param.path || param.key) || ""),
      filters: param.filters || [],
    });
    if (selected) setValue(param.path || param.key, selected);
  };

  return (
    <section className="main-grid config-center-layout">
      <div className="config-center-main">
        {filteredGroups.map((group) => {
          const isOpen = expandedGroups.includes(group.id);
          return (
            <article
              key={group.id}
              id={`config-group-${group.id}`}
              className="config-card config-group-card"
            >
              <button
                type="button"
                className="config-group-header"
                onClick={() => toggleGroup(group.id)}
              >
                <span className="group-header-left">
                  <span className="material-icons">{group.icon || "tune"}</span>
                  <span>
                    <strong>{group.title}</strong>
                    <small>{group.description}</small>
                  </span>
                </span>
                <span className="group-header-right">
                  <span className="group-count">{group.params.length} 项</span>
                  <span className="material-icons">
                    {isOpen ? "expand_less" : "expand_more"}
                  </span>
                </span>
              </button>

              {isOpen ? (
                <div className="config-group-body">
                  <div className="config-field-grid">
                    {(group.params || []).map((param) => {
                      const fieldPath = param.path || param.key;
                      const value = getValue(fieldPath);
                      const isHighlighted = highlightedKeys.has(fieldPath);
                      return (
                        <div
                          key={fieldPath}
                          className={`config-field-card ${
                            isHighlighted ? "highlighted" : ""
                          }`}
                        >
                          <ParameterField
                            param={param}
                            value={value}
                            onChange={(next) => setValue(fieldPath, next)}
                            onBrowse={
                              param.type === "path"
                                ? () => handleBrowse(param)
                                : undefined
                            }
                            disabled={running}
                          />
                          {param.hint ? (
                            <small className="config-field-hint">
                              {param.hint}
                            </small>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}
