import { useEffect, useState } from "react";
import { Activity, Bot, Moon, Settings2, Sun } from "lucide-react";
import newLogoDark from "../static/new_logo-dark.png";
import newLogoLight from "../static/new_logo-light.png";
import AgentPanel from "./features/agent/AgentPanel";
import MetricsPanel from "./features/metrics/MetricsPanel";
import TrainingPanel from "./features/training/TrainingPanel";
import { Button } from "./components/ui/button";
import { useToast } from "./hooks/useToast";

function App() {
  const [activeTab, setActiveTab] = useState("training");
  const [theme, setTheme] = useState(() => window.localStorage.getItem("edgedistilldet-theme") || "light");
  const { toasts, push } = useToast();
  const navItems = [
    { key: "training", icon: Settings2, label: "训练配置", desc: "配置蒸馏参数并启动训练流程" },
    { key: "advanced", icon: Settings2, label: "高级参数配置", desc: "集中配置训练与蒸馏高级参数" },
    { key: "metrics", icon: Activity, label: "指标监控", desc: "查看训练曲线与关键性能指标" },
    { key: "display", icon: Settings2, label: "可视化推理", desc: "配置推理与结果展示参数" },
    { key: "export", icon: Settings2, label: "导出模型", desc: "配置模型导出" },
    { key: "agent", icon: Bot, label: "Agent", desc: "通过智能助手分析与辅助调参" }
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
                className={`sidebar-nav-btn ${activeTab === item.key ? "active" : ""}`}
                onClick={() => setActiveTab(item.key)}
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
              onClick={() => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
            >
              {theme === "light" ? <Sun size={14} className="icon-sun" /> : <Moon size={14} className="icon-moon" />}
              <span className="theme-toggle-label">{theme === "light" ? "浅色模式" : "深色模式"}</span>
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
          </header>

          <div className="tab-panels console-panels">
            <TrainingPanel toast={push} active={activeTab === "training" || activeTab === "advanced" || activeTab === "display" || activeTab === "export"} view={activeTab} />
            <MetricsPanel toast={push} active={activeTab === "metrics"} />
            <AgentPanel toast={push} active={activeTab === "agent"} />
          </div>
        </main>
      </div>

      <div id="toast-container" className="toast-container">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            {toast.type === "success" ? <Activity size={16} /> : toast.type === "error" ? <Bot size={16} /> : <Settings2 size={16} />}
            <span>{toast.message}</span>
          </div>
        ))}
      </div>
    </>
  );
}

export default App;
