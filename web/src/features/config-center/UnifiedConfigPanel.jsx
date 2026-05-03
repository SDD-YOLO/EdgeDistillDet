import { useMemo, useState } from "react";
import { Button } from "../../components/ui/button";
import { ParameterField } from "../../components/forms/ParameterField";
import ConfigNav from "./ConfigNav";
import YamlPreview from "./YamlPreview";

export default function UnifiedConfigPanel({
  groups,
  form,
  getValue,
  setValue,
  pickLocalPath,
  previewPayload,
  running,
  isResumeMode,
  renderedHint,
  runHint,
  onSave,
  onLoad,
  onReset,
  onStartTraining,
  onStopTraining,
  onStartDisplay,
  onStopDisplay,
  onStartExport,
  onStopExport,
  trainingRunning,
  displayRunning,
  exportRunning,
}) {
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedGroups, setExpandedGroups] = useState(() =>
    groups.filter((g) => g.priority === "high").map((g) => g.id),
  );

  const filteredGroups = useMemo(() => {
    if (!searchQuery?.trim()) return groups;
    const query = searchQuery.toLowerCase();
    return groups
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
  }, [groups, searchQuery]);

  const highlightedKeys = useMemo(() => {
    if (!searchQuery?.trim()) return new Set();
    const query = searchQuery.toLowerCase();
    return new Set(
      groups
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
  }, [groups, searchQuery]);

  const toggleGroup = (groupId) => {
    setExpandedGroups((prev) =>
      prev.includes(groupId)
        ? prev.filter((id) => id !== groupId)
        : [...prev, groupId],
    );
  };

  const expandAll = () => setExpandedGroups(groups.map((group) => group.id));
  const collapseAll = () => setExpandedGroups([]);

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
    <section className="config-center-layout">
      <aside className="config-center-left">
        <div className="config-card config-center-toolbar">
          <div className="card-header">配置中心</div>
          <div className="config-toolbar-actions">
            <Button
              variant="default"
              className="btn-start"
              onClick={onStartTraining}
              disabled={
                trainingRunning || running || (isResumeMode && !trainingRunning)
              }
            >
              <span className="material-icons">play_arrow</span>
              开始训练
            </Button>
            <Button
              variant="destructive"
              className="btn-stop"
              onClick={onStopTraining}
              disabled={!trainingRunning}
            >
              <span className="material-icons">stop</span>
              停止训练
            </Button>
            <Button variant="outline" onClick={onSave} disabled={running}>
              <span className="material-icons">save</span>
              保存配置
            </Button>
            <Button variant="ghost" onClick={onLoad} disabled={running}>
              <span className="material-icons">file_open</span>
              加载配置
            </Button>
            <Button variant="ghost" onClick={onReset} disabled={running}>
              <span className="material-icons">restart_alt</span>
              重置表单
            </Button>
            <Button
              variant="outline"
              onClick={onStartDisplay}
              disabled={displayRunning}
            >
              <span className="material-icons">visibility</span>
              开始推理
            </Button>
            <Button
              variant="destructive"
              onClick={onStopDisplay}
              disabled={!displayRunning}
            >
              <span className="material-icons">stop</span>
              停止推理
            </Button>
            <Button
              variant="outline"
              onClick={onStartExport}
              disabled={exportRunning}
            >
              <span className="material-icons">file_upload</span>
              开始导出
            </Button>
            <Button
              variant="destructive"
              onClick={onStopExport}
              disabled={!exportRunning}
            >
              <span className="material-icons">stop</span>
              停止导出
            </Button>
          </div>
          <div className="config-toolbar-search">
            <input
              className="md-input"
              placeholder="搜索参数（名称、键值、说明）..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <div className="config-toolbar-actions compact">
              <Button size="sm" variant="outline" onClick={expandAll}>
                展开全部
              </Button>
              <Button size="sm" variant="outline" onClick={collapseAll}>
                折叠全部
              </Button>
            </div>
          </div>
          <div className="config-toolbar-hint">
            {renderedHint ||
              runHint ||
              "统一管理所有训练、蒸馏、导出与推理配置。"}
          </div>
        </div>

        <ConfigNav
          groups={filteredGroups}
          activeGroup={
            filteredGroups.find((g) => expandedGroups.includes(g.id))?.id ||
            filteredGroups[0]?.id ||
            ""
          }
          onNavigate={(groupId) => {
            const el = document.getElementById(`config-group-${groupId}`);
            if (el) {
              el.scrollIntoView({ behavior: "smooth", block: "start" });
              setExpandedGroups((prev) =>
                prev.includes(groupId) ? prev : [...prev, groupId],
              );
            }
          }}
        />
      </aside>

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

      <div className="config-center-right">
        <YamlPreview payload={previewPayload} />
      </div>
    </section>
  );
}
