"""RQ Worker 中执行的训练任务。"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import UUID

# 确保仓库根在 path
_WEB = Path(__file__).resolve().parent.parent
_REPO = _WEB.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def run_training_job(job_id: str) -> None:
    from web.core.paths import BASE_DIR
    from web.db.models import JobStatus, TrainingJob
    from web.db.session import SessionLocal, get_engine
    from web.saas.team_fs import ensure_team_workspace, team_root_path
    from web.core.workspace import reset_workspace, set_workspace

    get_engine()
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, job_id)
        if not job:
            return
        team_id = job.team_id
        ensure_team_workspace(team_id)
        root = team_root_path(team_id)
        log_dir = root / ".job_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job_id}.log"
        job.log_path = str(log_path)
        job.status = JobStatus.running.value
        job.started_at = datetime.utcnow()
        db.commit()

        t1, t2 = set_workspace(UUID(team_id), root)
        try:
            config_path = root / "configs" / job.config_name
            if not config_path.exists():
                raise FileNotFoundError(f"配置不存在: {config_path}")
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "scripts.train_with_distill",
                "--config",
                str(config_path),
            ]
            mode = job.mode or "distill"
            if mode == "resume":
                if job.checkpoint:
                    cp = Path(job.checkpoint)
                    if not cp.is_absolute():
                        cp = (root / cp).resolve()
                    cmd.extend(["--resume", str(cp)])
                else:
                    cmd.extend(["--resume", "auto"])
            elif job.allow_overwrite:
                cmd.append("--allow-overwrite")

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            env["EDGE_WEB_LOG"] = "1"

            with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
                logf.write(f"[job {job_id}] start {datetime.utcnow().isoformat()}\n")
                logf.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(BASE_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    text=True,
                    env=env,
                )
                assert proc.stdout is not None
                for line in iter(proc.stdout.readline, ""):
                    if not line and proc.poll() is not None:
                        break
                    logf.write(line)
                    logf.flush()
                proc.wait()
                rc = proc.returncode or 0
            job = db.get(TrainingJob, job_id)
            if job:
                job.finished_at = datetime.utcnow()
                job.status = JobStatus.succeeded.value if rc == 0 else JobStatus.failed.value
                if rc != 0:
                    job.error_message = f"进程退出码 {rc}"
                db.commit()
        finally:
            reset_workspace(t1, t2)
    except Exception as e:
        job = db.get(TrainingJob, job_id)
        if job:
            job.status = JobStatus.failed.value
            job.error_message = str(e)
            job.finished_at = datetime.utcnow()
            db.commit()
        raise
    finally:
        db.close()
