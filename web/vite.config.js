import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 与 FastAPI `mount("/static", StaticFiles(...))` 一致：产物中字体等资源须带此前缀，否则 CSS 会请求 /assets/*.woff2 导致 404，图标退化为文字
export default defineConfig(({ command }) => ({
  base: command === "build" ? "/static/dist/" : "/",
  plugins: [react()],
  test: {
    environment: "jsdom",
    clearMocks: true,
    restoreMocks: true
  },
  // 开发时前端在 5173，API 在 FastAPI（默认 127.0.0.1:5000），用代理避免跨域
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:5000", changeOrigin: true },
      "/static": { target: "http://127.0.0.1:5000", changeOrigin: true },
      "/favicon.ico": { target: "http://127.0.0.1:5000", changeOrigin: true },
    },
  },
  build: {
    outDir: "static/dist",
    emptyOutDir: true,
    rollupOptions: {
      input: "src/main.jsx",
      output: {
        entryFileNames: "app.js",
        chunkFileNames: "chunks/[name].js",
        assetFileNames: (assetInfo) => {
          if (assetInfo.name && assetInfo.name.endsWith(".css")) {
            return "app.css";
          }
          // Use hashed asset filenames to avoid stale font/icon caches.
          return "assets/[name]-[hash][extname]";
        }
      }
    }
  }
}));
