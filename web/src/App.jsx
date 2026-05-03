import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { Activity, Bot, Moon, Settings2, Sun } from "lucide-react";
import newLogoDark from "../static/new_logo-dark.png";
import newLogoLight from "../static/new_logo-light.png";
import { Button } from "./components/ui/button";
import { useToast } from "./hooks/useToast";
import { CONFIG_GROUPS } from "./constants/configGroups";
import { useTrainingPanelController } from "./features/training/hooks/useTrainingPanelController";

const TrainingPanel = lazy(() => import("./features/training/TrainingPanel"));
const MetricsPanel = lazy(() => import("./features/metrics/MetricsPanel"));
const AgentPanel = lazy(() => import("./features/agent/AgentPanel"));

function App() {
  const [activeTab, setActiveTab] = useState("config-center");
  const [theme, setTheme] = useState(
    () => window.localStorage.getItem("edgedistilldet-theme") || "light",
  );
  const { toasts, push } = useToast();
  const navItems = [
    {
      key: "config-center",
      icon: Settings2,
      label: "配置中心",
      desc: "统一管理训练、蒸馏、导出与推理参数",
    },
    {
      key: "metrics",
      icon: Activity,
      label: "指标监控",
      desc: "查看训练曲线与关键性能指标",
    },
    {
      key: "agent",
      icon: Bot,
      label: "Agent",
      desc: "通过智能助手分析与辅助调参",
    },
  ];
  const activeNav =
    navItems.find((item) => item.key === activeTab) || navItems[0];

  const controller = useTrainingPanelController({ toast: push });
  const [configActiveGroup, setConfigActiveGroup] = useState(
    CONFIG_GROUPS[0]?.id || "",
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedGroups, setExpandedGroups] = useState(() =>
    CONFIG_GROUPS.filter((g) => g.priority === "high").map((g) => g.id),
  );

  const allGroupIds = useMemo(() => CONFIG_GROUPS.map((group) => group.id), []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("edgedistilldet-theme", theme);
  }, [theme]);

  return (
    <>
      <div className="console-shell">
        <aside className="console-sidebar">
          <div className="sidebar-brand">
            <img
              src={theme === "dark" ? newLogoDark : newLogoLight}
              alt="EdgeDistillDet Logo"
              width={42}
              height={42}
              style={{ borderRadius: 10 }}
            />
            <div className="brand-text">
              <h1 className="app-title">EdgeDistillDet</h1>
              <span className="app-subtitle">边缘蒸馏训练工作台</span>
            </div>
          </div>

          <div className="status-indicator online">
            <span className="status-dot" />
            <span>服务正常</span>
          </div>

          <nav className="sidebar-nav" role="tablist" aria-label="主导航">
            {navItems.map((item) => (
              <Button
                key={item.key}
                className={`sidebar-nav-btn ${
                  activeTab === item.key ? "active" : ""
                }`}
                onClick={() => {
                  setActiveTab(item.key);
                }}
                variant="ghost"
                legacy={false}
              >
                <item.icon size={16} />
                <span>{item.label}</span>
              </Button>
            ))}
          </nav>

          <div className="sidebar-bottom">
            <Button
              variant="outline"
              size="sm"
              className="theme-toggle"
              title="切换明暗主题"
              onClick={() =>
                setTheme((prev) => (prev === "light" ? "dark" : "light"))
              }
            >
              {theme === "light" ? (
                <Sun size={14} className="icon-sun" />
              ) : (
                <Moon size={14} className="icon-moon" />
              )}
              <span className="theme-toggle-label">
                {theme === "light" ? "浅色模式" : "深色模式"}
              </span>
            </Button>
          </div>
        </aside>

        <main className="console-main">
          <header className="console-page-header">
            <div className="console-page-meta">
              <activeNav.icon size={18} className="text-primary" />
              <div>
                <h2>{activeNav.label}</h2>
                <p>{activeNav.desc}</p>
              </div>
            </div>
            {activeTab === "config-center" ? (
              <div className="config-center-toolbar-inline">
                <Button
                  variant="ghost"
                  onClick={controller.startTraining}
                  disabled={controller.running}
                >
                  <span className="material-icons">play_arrow</span>
                  开始训练
                </Button>
                <Button
                  variant="destructive"
                  onClick={controller.stopTraining}
                  disabled={!controller.running}
                >
                  <span className="material-icons">stop</span>
                  停止训练
                </Button>
                <Button
                  variant="outline"
                  onClick={async () => {
                    try {
                      await controller.saveConfig();
                      push("配置已保存", "success");
                    } catch (error) {
                      push(error.message, "error");
                    }
                  }}
                  disabled={controller.running}
                >
                  <span className="material-icons">save</span>
                  保存配置
                </Button>
                <Button
                  variant="ghost"
                  onClick={controller.loadConfigFromFile}
                  disabled={controller.running}
                >
                  <span className="material-icons">file_open</span>
                  加载配置
                </Button>
                <Button
                  variant="ghost"
                  onClick={controller.resetForm}
                  disabled={controller.running}
                >
                  <span className="material-icons">restart_alt</span>
                  重置表单
                </Button>
                <Button
                  variant="outline"
                  onClick={controller.startInference}
                  disabled={controller.inferRunning}
                >
                  <span className="material-icons">visibility</span>
                  开始推理
                </Button>
                <Button
                  variant="destructive"
                  onClick={controller.stopInference}
                  disabled={!controller.inferRunning}
                >
                  <span className="material-icons">stop</span>
                  停止推理
                </Button>
                <Button
                  variant="outline"
                  onClick={controller.startExport}
                  disabled={controller.exportRunning}
                >
                  <span className="material-icons">file_upload</span>
                  开始导出
                </Button>
                <Button
                  variant="destructive"
                  onClick={controller.stopExport}
                  disabled={!controller.exportRunning}
                >
                  <span className="material-icons">stop</span>
                  停止导出
                </Button>
                <div className="config-toolbar-search inline">
                  <input
                    className="md-input"
                    placeholder="搜索参数（名称、键值、说明）..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  <div className="config-toolbar-actions compact">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setExpandedGroups(allGroupIds)}
                    >
                      展开全部
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setExpandedGroups([])}
                    >
                      折叠全部
                    </Button>
                  </div>
                </div>
                <div className="config-toolbar-hint inline">
                  {controller.renderedHint ||
                    controller.runHint ||
                    "统一管理所有训练、蒸馏、导出与推理配置。"}
                </div>
              </div>
            ) : null}
          </header>

          <Suspense
            fallback={
              <div className="tab-panels console-panels">
                <div className="panel-loading">正在加载面板...</div>
              </div>
            }
          >
            <div className="tab-panels console-panels">
              <TrainingPanel
                toast={push}
                active={activeTab === "config-center"}
                view={activeTab}
                groups={CONFIG_GROUPS}
                configActiveGroup={configActiveGroup}
                onConfigNavigate={(g) => setConfigActiveGroup(g)}
                controller={controller}
                searchQuery={searchQuery}
                expandedGroups={expandedGroups}
                setExpandedGroups={setExpandedGroups}
              />
              <MetricsPanel toast={push} active={activeTab === "metrics"} />
              <AgentPanel toast={push} active={activeTab === "agent"} />
            </div>
          </Suspense>
        </main>
      </div>

      <div id="toast-container" className="toast-container">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            {toast.type === "success" ? (
              <Activity size={16} />
            ) : toast.type === "error" ? (
              <Bot size={16} />
            ) : (
              <Settings2 size={16} />
            )}
            <span>{toast.message}</span>
          </div>
        ))}
      </div>
    </>
  );
}

export default App;
