# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的约定，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
