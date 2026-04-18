# 重构前回归基线（行为快照）

本文档记录结构重构前的关键行为快照，用于重构后逐项对比，确保“不改功能”约束成立。

## 环境

- 项目根目录：`d:/Personal_Files/Projects/EdgeDistillDet`
- Web 服务：`python web/app.py`
- 访问地址：`http://127.0.0.1:5000`

## CLI 快照

执行命令：`python main.py --help`

关键输出（节选）：

- 程序名：`EdgeDistillDet`
- 子命令：`train` / `eval` / `analyze` / `profile`
- 帮助页可正常显示，无异常退出

## API 快照

### `GET /api/configs`

结果（节选）：

```json
{
  "status": "ok",
  "configs": [
    "_test_resume.yaml",
    "agent_prompts.yaml",
    "dataset_coco.yaml",
    "dataset_coco128.yaml",
    "dataset_custom_template.yaml",
    "dataset_mini_test.yaml",
    "dataset_visdrone.yaml",
    "distill_config.yaml",
    "distill_config_cloud_example.yaml",
    "distill_config_remote_api_example.yaml",
    "eval_config.yaml"
  ]
}
```

### `GET /api/train/status`

结果：

```json
{
  "status": "ok",
  "running": false,
  "pid": null,
  "config": null,
  "mode": "distill",
  "start_time": null,
  "current_epoch": 0,
  "total_epochs": 0,
  "logs": [],
  "log_count": 0
}
```

### `GET /api/metrics`

结果（节选）：

```json
{
  "status": "ok",
  "csv_metrics": [
    {
      "name": "exp2",
      "display_name": "exp2 @ 2026-04-18 10:10:31",
      "dir": "runs\\detect\\runs\\exp2",
      "has_results": true,
      "path": "runs\\detect\\runs\\exp2\\results.csv",
      "rows": 10
    }
  ]
}
```

## 回归判定

重构后需满足：

1. 上述三个 API 路径、顶层字段与语义保持一致。
2. `main.py --help` 仍可显示四个子命令且无异常。
3. 训练状态接口在空闲时仍返回同等字段集（允许日志内容变化，不允许字段缺失）。
