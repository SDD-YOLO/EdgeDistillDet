import React from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/plus-jakarta-sans/latin-400.css";
import "@fontsource/plus-jakarta-sans/latin-500.css";
import "@fontsource/plus-jakarta-sans/latin-600.css";
import "@fontsource/plus-jakarta-sans/latin-700.css";
import "@fontsource/noto-sans-sc/chinese-simplified-400.css";
import "@fontsource/noto-sans-sc/chinese-simplified-500.css";
import "@fontsource/noto-sans-sc/chinese-simplified-700.css";
import "@fontsource/material-icons/400.css";
import App from "./App";
import "../static/css/styles.css";
import "./styles/index.css";

// #region agent log
window.addEventListener("error", (event) => {
  fetch("http://127.0.0.1:7934/ingest/2c4bcf68-efd6-4fd1-8130-1f5a368246bc", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "e872f3" },
    body: JSON.stringify({
      sessionId: "e872f3",
      runId: `runtime-${Date.now()}`,
      hypothesisId: "H66",
      location: "main.jsx:window.error",
      message: "uncaught runtime error",
      data: {
        message: String(event?.message || ""),
        filename: String(event?.filename || ""),
        lineno: Number(event?.lineno || 0),
        colno: Number(event?.colno || 0)
      },
      timestamp: Date.now()
    })
  }).catch(() => {});
});

window.addEventListener("unhandledrejection", (event) => {
  fetch("http://127.0.0.1:7934/ingest/2c4bcf68-efd6-4fd1-8130-1f5a368246bc", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "e872f3" },
    body: JSON.stringify({
      sessionId: "e872f3",
      runId: `runtime-${Date.now()}`,
      hypothesisId: "H67",
      location: "main.jsx:window.unhandledrejection",
      message: "unhandled promise rejection",
      data: {
        reason:
          typeof event?.reason === "string"
            ? event.reason
            : String(event?.reason?.message || event?.reason || "")
      },
      timestamp: Date.now()
    })
  }).catch(() => {});
});

window.addEventListener("beforeunload", () => {
  fetch("http://127.0.0.1:7934/ingest/2c4bcf68-efd6-4fd1-8130-1f5a368246bc", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "e872f3" },
    body: JSON.stringify({
      sessionId: "e872f3",
      runId: `runtime-${Date.now()}`,
      hypothesisId: "H72",
      location: "main.jsx:beforeunload",
      message: "page is unloading",
      data: { href: String(window.location.href || "") },
      timestamp: Date.now()
    })
  }).catch(() => {});
});

window.addEventListener("pagehide", () => {
  fetch("http://127.0.0.1:7934/ingest/2c4bcf68-efd6-4fd1-8130-1f5a368246bc", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "e872f3" },
    body: JSON.stringify({
      sessionId: "e872f3",
      runId: `runtime-${Date.now()}`,
      hypothesisId: "H72",
      location: "main.jsx:pagehide",
      message: "pagehide fired",
      data: { href: String(window.location.href || "") },
      timestamp: Date.now()
    })
  }).catch(() => {});
});

const rootEl = document.getElementById("root");
if (rootEl) {
  const observer = new MutationObserver(() => {
    const hasShell = !!document.querySelector(".console-shell");
    if (!hasShell || rootEl.childElementCount === 0) {
      fetch("http://127.0.0.1:7934/ingest/2c4bcf68-efd6-4fd1-8130-1f5a368246bc", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "e872f3" },
        body: JSON.stringify({
          sessionId: "e872f3",
          runId: `runtime-${Date.now()}`,
          hypothesisId: "H73",
          location: "main.jsx:rootMutation",
          message: "root content missing or app shell absent",
          data: {
            childCount: Number(rootEl.childElementCount || 0),
            hasShell
          },
          timestamp: Date.now()
        })
      }).catch(() => {});
    }
  });
  observer.observe(rootEl, { childList: true, subtree: true });
}
// #endregion

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
