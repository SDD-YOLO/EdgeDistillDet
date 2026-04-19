import { useEffect, useState } from "react";
import AgentPanel from "./features/agent/AgentPanel";
import MetricsPanel from "./features/metrics/MetricsPanel";
import TrainingPanel from "./features/training/TrainingPanel";
import { useToast } from "./hooks/useToast";
import SaasBar from "./components/SaasBar";

function App() {
  const [activeTab, setActiveTab] = useState("training");
  const [theme, setTheme] = useState(() => window.localStorage.getItem("edgedistilldet-theme") || "light");
  const { toasts, push } = useToast();
  const navItems = [
    { key: "training", icon: "tune", label: "训练配置", desc: "配置蒸馏参数并启动训练流程" },
    { key: "metrics", icon: "analytics", label: "指标监控", desc: "查看训练曲线与关键性能指标" },
    { key: "agent", icon: "smart_toy", label: "Agent", desc: "通过智能助手分析与辅助调参" }
  ];
  const activeNav = navItems.find((item) => item.key === activeTab) || navItems[0];

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("edgedistilldet-theme", theme);
  }, [theme]);

  return (
    <>
      <div className="console-shell">
        <aside className="console-sidebar">
          <div className="sidebar-brand">
            <span className="material-icons app-icon">radar</span>
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
              <button
                key={item.key}
                className={`sidebar-nav-btn ${activeTab === item.key ? "active" : ""}`}
                onClick={() => setActiveTab(item.key)}
              >
                <span className="material-icons tab-icon">{item.icon}</span>
                <span>{item.label}</span>
              </button>
            ))}
          </nav>

          <div className="sidebar-bottom">
            <button
              className="theme-toggle"
              title="切换明暗主题"
              onClick={() => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
            >
              <span className="material-icons icon-sun">light_mode</span>
              <span className="material-icons icon-moon">dark_mode</span>
              <span className="theme-toggle-label">{theme === "light" ? "浅色模式" : "深色模式"}</span>
            </button>
          </div>
        </aside>

        <main className="console-main">
          <header className="console-page-header">
            <div className="console-page-meta">
              <span className="material-icons">{activeNav.icon}</span>
              <div>
                <h2>{activeNav.label}</h2>
                <p>{activeNav.desc}</p>
              </div>
            </div>
            <SaasBar toast={push} />
          </header>

          <div className="tab-panels console-panels">
            <TrainingPanel toast={push} active={activeTab === "training"} />
            <MetricsPanel toast={push} active={activeTab === "metrics"} />
            <AgentPanel toast={push} active={activeTab === "agent"} />
          </div>
        </main>
      </div>

      <div id="toast-container" className="toast-container">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            <span className="material-icons">
              {toast.type === "success" ? "check_circle" : toast.type === "error" ? "error" : "info"}
            </span>
            <span>{toast.message}</span>
          </div>
        ))}
      </div>
    </>
  );
}

export default App;
