"""SaaS 训练任务入队（Redis RQ）。"""

from __future__ import annotations

from pathlib import Path

from web.core.paths import get_config_dir
from web.core.request_context import get_request_user_id
from web.core.saas_settings import get_redis_url, use_training_queue
from web.core.workspace import get_current_team_id
from web.db.models import JobStatus, TrainingJob
from web.db.session import SessionLocal, get_engine
from web.schemas import TrainStartRequest
from web.services.backend_common import _load_yaml_file, _normalize_compute_provider


def maybe_enqueue_training(payload: TrainStartRequest) -> dict | None:
    if not use_training_queue():
        return None
    team_id = get_current_team_id()
    user_id = get_request_user_id()
    if not team_id or not user_id:
        return None
    cfg_path = get_config_dir() / payload.config
    cfg = _load_yaml_file(cfg_path) or {}
    train_cfg = dict(cfg.get("training", {}) or {})
    if _normalize_compute_provider(train_cfg.get("compute_provider")) == "remote_api":
        return None

    get_engine()
    db = SessionLocal()
    try:
        job = TrainingJob(
            team_id=str(team_id),
            created_by=user_id,
            status=JobStatus.queued.value,
            config_name=payload.config,
            mode=payload.mode,
            checkpoint=payload.checkpoint,
            allow_overwrite=bool(payload.allow_overwrite),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        jid = job.id
    finally:
        db.close()

    try:
        from redis import Redis
        from rq import Queue

        q = Queue("training", connection=Redis.from_url(get_redis_url()))
        rq_job = q.enqueue(
            "web.jobs.training_job_worker.run_training_job",
            jid,
            job_timeout=-1,
        )
        db = SessionLocal()
        try:
            j = db.get(TrainingJob, jid)
            if j:
                j.rq_job_id = rq_job.id
                db.commit()
        finally:
            db.close()
    except Exception as e:
        db = SessionLocal()
        try:
            j = db.get(TrainingJob, jid)
            if j:
                j.status = JobStatus.failed.value
                j.error_message = str(e)
                db.commit()
        finally:
            db.close()
        return {"status": "error", "message": f"入队失败: {e}"}

    return {
        "status": "ok",
        "message": "训练任务已加入队列",
        "queued": True,
        "job_id": jid,
    }


def get_team_training_status_snapshot() -> dict | None:
    """队列模式下返回与前端兼容的训练状态（含日志尾部）。"""
    if not use_training_queue():
        return None
    tid = get_current_team_id()
    if not tid:
        return None
    job = get_active_job_for_team(str(tid))
    if not job:
        return None
    logs: list[str] = []
    if job.log_path and Path(job.log_path).exists():
        try:
            text = Path(job.log_path).read_text(encoding="utf-8", errors="replace")
            logs = text.splitlines()[-500:]
        except OSError:
            pass
    running = job.status in (JobStatus.queued.value, JobStatus.running.value)
    return {
        "status": "ok",
        "running": running,
        "pid": None,
        "config": job.config_name,
        "mode": job.mode,
        "start_time": job.started_at.timestamp() if job.started_at else None,
        "current_epoch": 0,
        "total_epochs": 0,
        "logs": logs,
        "log_count": len(logs),
        "job_id": job.id,
        "queue_status": job.status,
        "error": job.error_message,
    }


def get_active_job_for_team(team_id: str) -> TrainingJob | None:
    get_engine()
    db = SessionLocal()
    try:
        from sqlalchemy import select

        row = db.scalars(
            select(TrainingJob)
            .where(
                TrainingJob.team_id == team_id,
                TrainingJob.status.in_([JobStatus.queued.value, JobStatus.running.value]),
            )
            .order_by(TrainingJob.created_at.desc())
        ).first()
        return row
    finally:
        db.close()
