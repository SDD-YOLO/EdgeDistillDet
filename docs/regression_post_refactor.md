# 重构后回归结果

## 执行项

1. Python 语法编译检查（核心改动文件）
2. 前端构建检查（`npm run build`）
3. CLI 冒烟检查（`python main.py --help`）
4. API 行为快照对比：
   - `GET /api/configs`
   - `GET /api/train/status`
   - `GET /api/metrics`

## 结果

- Python 编译：通过
- 前端构建：通过（Vite 成功产出 `web/static/dist/app.js`、`web/static/dist/app.css`）
- CLI 帮助：通过（子命令仍为 `train/eval/analyze/profile`）
- API 快照：通过（关键路径与字段保持一致）

## 与基线对比结论

与 `docs/regression_baseline.md` 对比：

- 路由路径保持一致
- 顶层字段与语义保持一致
- 空闲训练状态结构保持一致
- 指标列表接口可正常返回

结论：本次重构未引入已观测到的功能性回归。
