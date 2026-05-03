import UnifiedConfigPanel from "../config-center/UnifiedConfigPanel";
import ConfigNav from "../config-center/ConfigNav";

function TrainingPanel({
  toast,
  active,
  view = "training",
  groups,
  configActiveGroup,
  onConfigNavigate,
  controller,
  searchQuery,
  expandedGroups,
  setExpandedGroups,
}) {
  return (
    <div
      className={`tab-panel console-module-panel ${active ? "active" : ""}`}
      id="panel-training"
      aria-hidden={!active}
    >
      <section className="config-center-shell">
        <aside className="config-group-selector">
          <ConfigNav
            groups={groups}
            activeGroup={configActiveGroup}
            onNavigate={(groupId) => {
              setExpandedGroups([groupId]);
              onConfigNavigate(groupId);
            }}
          />
        </aside>

        <UnifiedConfigPanel
          groups={groups}
          activeGroup={configActiveGroup}
          onNavigate={onConfigNavigate}
          form={controller.form}
          getValue={controller.getValueByPath}
          setValue={controller.setValueByPath}
          pickLocalPath={controller.pickLocalPath}
          running={controller.running}
          isResumeMode={false}
          searchQuery={searchQuery}
          expandedGroups={expandedGroups}
          setExpandedGroups={setExpandedGroups}
        />
      </section>
    </div>
  );
}
export default TrainingPanel;
