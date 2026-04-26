# Changelog

## [1.0.5] - 2026-04-26

### Changed

- 更新readme.md文档

## [1.0.4] - 2026-04-26

### Changed

- 优化设备检测模块，自动检测可用设备并选择最优设备

## [1.0.3] - 2026-04-25

### Changed

- 优化agentprompt，强制agent在提出参数修改的时候调用preview工具触发审核
- 优化agent对于复杂任务的工具链拆解
- 去除agent.propose_patch，agent.validate_patch等工具冗余
- 修复start_web没有构建前端的bug
- 修复训练产物的路径问题
- 修复agent输出的气泡中json代码未被清洗的bug

## [1.0.2] - 2026-04-25

### Changed

- 移除前端模板中的调试埋点上报脚本（`/ingest`、`X-Debug-Session-Id`、`sessionId/runId/hypothesisId`）。
- 清理仓库中的历史调试埋点日志文件（`debug-*.log` 与 `.cursor/debug-*.log`）。
- 文档同步：明确当前版本默认不包含前端埋点上报逻辑。

## [1.0.1] - 2026-04-18

### Added

- `GET /api/version`：返回与 `main.__version__` 一致的包版本信息。
- FastAPI OpenAPI 元数据中的 `version` 与 Python 包版本对齐。

### Changed

- 以 `main.__version__` 为唯一版本来源；`pyproject.toml` 通过 setuptools `dynamic` 读取该属性。
- 根目录 `__init__.py` 从 `main` 再导出 `__version__`。
- 完善 [README.md](README.md)：完整 HTTP API 列表、Node 20 与 `engines` 说明、版本/许可/发版检查清单。
- `web/package.json` 与根包版本对齐为 1.0.1。

## [1.0.0] - 2026-04-18

### Added

- 初始公开发布：CLI（train / eval / analyze / profile）、蒸馏与评估流程、本地 Web 工作台（FastAPI + React）、配置驱动与 Agent 相关接口。
