from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
import urllib.error
from pathlib import Path

from fastapi import Query
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from web.core.paths import BASE_DIR, CONFIG_DIR
from web.schemas import TrainStartRequest
from web.services import training_runtime
from web.services.backend_common import _candidate_output_roots, _error, _list_resume_candidates, _load_yaml_file, _normalize_compute_provider, _resolve_project_path
from web.services.backend_train_runtime import (
    _TRAIN_LOCK_FILE,
    _acquire_training_lock,
    _build_cloud_api_config,
    _cleanup_gpu_resources,
    _http_json_request,
    _is_process_alive,
    _kill_old_training,
    _kill_process_tree,
    _release_os_file_lock,
    _release_training_lock,
    _resolve_dataset_via_api,
    _run_training_process_safe,
    _scan_and_kill_stale_training_processes,
    _update_status_line,
    _wait_for_gpu_free,
)

def output_check(project: str = Query('runs/distill')):
    project = project or 'runs/distill'
    try:
        project_path = _resolve_project_path(project, allow_external=Path(project).is_absolute())
    except ValueError as e:
        return _error(str(e), 400)

    existing_names = []
    next_exp = 'exp1'
    candidate_roots = _candidate_output_roots(project_path)
    merged_names = set()
    for root in candidate_roots:
        if root.exists() and root.is_dir():
            for item in sorted(root.iterdir()):
                if item.is_dir():
                    merged_names.add(item.name)
    existing_names = sorted(merged_names)

    exp_numbers = []
    for name in existing_names:
        if name.startswith('exp'):
            try:
                exp_numbers.append(int(name[3:] or 0))
            except ValueError:
                pass
    next_exp = f'exp{max(exp_numbers) + 1}' if exp_numbers else 'exp1'
    return {
        'status': 'ok',
        'project': str(project_path.relative_to(BASE_DIR)),
        'existing_names': existing_names,
        'next_exp_name': next_exp,
    }

def start_training(payload: TrainStartRequest):
    """
    启动训练 — 五层防护，坚决杜绝双进程同时运行
    
    ═══════ 防护层次（由外到内）══════
      Layer-0: 线程级互斥锁     → 防 ASGI/多线程并发穿透
      Layer-1: 残留进程扫描     → 防重启后丢失引用的僵尸进程
      Layer-2: 内存状态检查     → 防 running=True 的已知进程逃逸  
      Layer-3: OS 级文件锁       → 防多实例/多 Web 进程并发（内核原子保证）
      Layer-4: GPU 安全等待     → 确保显存真正释放后才启动
    """

    config_name = payload.config
    mode = payload.mode
    checkpoint = payload.checkpoint
    allow_overwrite = bool(payload.allow_overwrite)
    if mode not in {'distill', 'resume'}:
        return _error(f'不支持的训练模式: {mode}', 400)

    # ════════════════════════════════════════════════════════
    # Layer-0：线程级互斥锁 — 防止并发请求穿透（uvicorn 线程池 / 多 worker 场景）
    # ════════════════════════════════════════════════════════
    acquired = training_runtime.train_thread_lock.acquire(blocking=False)
    if not acquired:
        return _error('系统繁忙：另一个请求正在处理中，请稍后再试', 503)

    try:
        # ═══════════════════════════════════════════════════
        # Layer-1：扫描残留/僵尸训练进程（Web 重启后丢失引用的场景）
        # ═══════════════════════════════════════════════════
        our_pid = os.getpid()
        stale = _scan_and_kill_stale_training_processes(exclude_pid=our_pid)
        if stale['found'] > 0:
            _update_status_line(f"[GUARD] 发现 {stale['found']} 个残留训练进程，已清理: {stale['details']}")
            time.sleep(3)

        # ═══════════════════════════════════════════════════
        # Layer-2：杀掉内存中已知的旧训练进程（resume 模式允许覆盖）
        # ═══════════════════════════════════════════════════
        with training_runtime.train_state_lock:
            busy = training_runtime.training_status['running']
        if busy or training_runtime.training_process is not None:
            if mode == 'resume':
                _update_status_line('[RESUME] 检测到旧训练进程，正在终止...')
                _kill_old_training()
                time.sleep(3)
                _cleanup_gpu_resources()
                time.sleep(1)
            else:
                return _error('已有训练任务在运行中，请先停止或等待完成', 400)

        # ═══════════════════════════════════════════════════
        # Layer-3：获取 OS 级别文件排他锁（内核原子保证，非模拟！）
        # ═══════════════════════════════════════════════════
        lock_ok = _acquire_training_lock()
        if not lock_ok:
            err_msg = '训练互斥锁被占用'
            try:
                if _TRAIN_LOCK_FILE.exists():
                    content = _TRAIN_LOCK_FILE.read_text().strip().split('\n')
                    holder_pid = int(content[0]) if content else -1
                    if holder_pid > 0:
                        alive = _is_process_alive(holder_pid)
                        if not alive:
                            _release_os_file_lock()
                            _TRAIN_LOCK_FILE.unlink(missing_ok=True)
                            lock_ok = _acquire_training_lock()
                            if lock_ok:
                                _update_status_line(f'[GUARD] 已清除僵尸锁 (原PID={holder_pid})，重新获取成功')
                        else:
                            err_msg = f'训练进程 (PID={holder_pid}) 仍在运行中'
                    else:
                        err_msg = '训练锁文件损坏'
            except Exception:
                pass
            
            if not lock_ok:
                return _error(err_msg, 400)

        # ═══════════════════════════════════════════════════
        # Layer-4：二次验证 + GPU 安全等待
        # ═══════════════════════════════════════════════════
        
        # 二次验证：再次扫描残留（防御性编程）
        recheck_stale = _scan_and_kill_stale_training_processes(exclude_pid=our_pid)
        if recheck_stale['found'] > 0:
            _update_status_line(f'[GUARD] 二次扫描发现 {recheck_stale["found"]} 个漏网进程，已清理')
            time.sleep(2)

        # 验证配置文件
        config_path = CONFIG_DIR / config_name
        if not config_path.exists():
            _release_training_lock()
            return _error(f'配置文件不存在: {config_name}', 404)

        cfg = _load_yaml_file(config_path) or {}
        train_cfg = dict(cfg.get('training', {}) or {})
        compute_provider = _normalize_compute_provider(train_cfg.get('compute_provider'))
        if compute_provider == 'remote_api':
            try:
                api_cfg = _build_cloud_api_config(train_cfg)
            except ValueError as e:
                return _error(str(e), 400)
            request_payload = {
                'config': cfg,
                'mode': mode,
                'checkpoint': checkpoint,
                'allow_overwrite': allow_overwrite,
            }
            try:
                dataset_result = _resolve_dataset_via_api(train_cfg, cfg, mode, checkpoint)
                if isinstance(dataset_result, dict):
                    dataset_yaml = str(dataset_result.get('data_yaml', '') or '').strip()
                    dataset_id = str(dataset_result.get('dataset_id', '') or '').strip()
                    if dataset_yaml:
                        request_payload['config'] = copy.deepcopy(cfg)
                        request_payload['config'].setdefault('training', {})
                        request_payload['config']['training']['data_yaml'] = dataset_yaml
                    request_payload['dataset'] = dataset_result
                    _update_status_line(
                        f"[REMOTE] 数据集API已解析: "
                        f"{dataset_yaml or dataset_id or 'unknown'}"
                    )
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode('utf-8', errors='replace')
                except Exception:
                    body = ''
                return _error(f'数据集 API 调用失败: HTTP {e.code} {body}', 502)
            except Exception as e:
                return _error(f'数据集 API 调用失败: {e}', 502)
            try:
                submit_result = _http_json_request('POST', api_cfg['submit_url'], payload=request_payload, headers=api_cfg['headers'])
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode('utf-8', errors='replace')
                except Exception:
                    body = ''
                return _error(f'云训练提交失败: HTTP {e.code} {body}', 502)
            except Exception as e:
                return _error(f'云训练提交失败: {e}', 502)

            job_id = str(submit_result.get('job_id', '') or submit_result.get('id', '') or '').strip()
            if not job_id:
                return _error('云训练接口未返回 job_id', 502)

            with training_runtime.train_state_lock:
                training_runtime.training_status.update({
                    'running': True,
                    'pid': None,
                    'config': config_name,
                    'mode': mode,
                    'start_time': time.time(),
                    'current_epoch': 0,
                    'total_epochs': 0,
                    'logs': [f"[REMOTE] 已提交云训练任务: job_id={job_id}"],
                })
                training_runtime.remote_training_state.update({
                    'active': True,
                    'job_id': job_id,
                    'api_base_url': api_cfg['base_url'],
                    'logs_offset': 0,
                })

            threading.Thread(target=_remote_polling_loop, args=(api_cfg, job_id), daemon=True).start()
            return {'status': 'ok', 'message': '云训练任务已提交', 'remote': True, 'job_id': job_id}
        allow_external_project = compute_provider in {'autodl', 'colab'}
        output_cfg = dict(cfg.get('output', {}) or {})
        target_project = str(output_cfg.get('project', 'runs/distill') or 'runs/distill')
        target_name = str(output_cfg.get('name', 'exp') or 'exp').strip()
        try:
            project_path = _resolve_project_path(target_project, allow_external=allow_external_project)
        except ValueError:
            _release_training_lock()
            return _error(f'输出目录非法: {target_project}', 400)
        candidate_roots = _candidate_output_roots(project_path)
        target_run_paths = [(root / target_name).resolve() for root in candidate_roots]
        existing_target_path = next((p for p in target_run_paths if p.exists()), None)

        if mode != 'resume' and target_name and existing_target_path is not None and not allow_overwrite:
            _release_training_lock()
            conflict_project = target_project
            try:
                conflict_project = str(existing_target_path.parent.relative_to(BASE_DIR))
            except Exception:
                pass
            return JSONResponse(status_code=409, content={
                'error': f'输出目录已存在：{conflict_project}/{target_name}',
                'requires_confirmation': True,
                'project': conflict_project,
                'name': target_name,
            })

        # 构建命令：蒸馏 / 断点续训 共用同一子进程入口，避免双栈逻辑与重复加载
        cmd = [sys.executable, '-u', '-m', 'scripts.train_with_distill', '--config', str(config_path)]
        if mode == 'resume':
            if checkpoint:
                checkpoint_path = Path(checkpoint)
                if not checkpoint_path.is_absolute():
                    checkpoint_path = (BASE_DIR / checkpoint_path).resolve()
                cmd.extend(['--resume', str(checkpoint_path)])
            else:
                cmd.append('--resume')
                cmd.append('auto')
        elif allow_overwrite:
            cmd.append('--allow-overwrite')
        # GPU 安全等待：确保显存真正释放后再启动新训练
        _cleanup_gpu_resources()
        gpu_ready = _wait_for_gpu_free(timeout_sec=20.0)
        if not gpu_ready:
            _release_training_lock()
            return JSONResponse(status_code=503, content={
                'error': 'GPU 显存未能在超时时间内释放，可能仍有残留训练进程',
                'hint': '请手动结束占用 GPU 的进程后重试',
            })

        # 最终步骤：更新状态并启动新训练线程
        with training_runtime.train_state_lock:
            training_runtime.training_status.update({
                'running': True,
                'pid': None,
                'config': config_name,
                'mode': mode,
                'start_time': time.time(),
                'current_epoch': 0,
                'total_epochs': 0,
                'logs': [f"{'[RESUME] 断点续训' if mode == 'resume' else '[TRAIN] 训练'} 已启动..."],
            })
        thread = threading.Thread(target=_run_training_process_safe, args=(cmd,), daemon=True)
        thread.start()
        return {'status': 'ok', 'message': f"{'断点续训' if mode == 'resume' else '训练'}已启动"}

    finally:
        # 【关键】线程锁在 finally 中释放。
        # 文件锁仍由子线程 (_run_training_process_safe) 持有直到训练结束。
        training_runtime.train_thread_lock.release()

def stop_training():
    """
    停止训练进程 — 彻底版
    
    修复：解决多线程竞态条件导致进程残留的问题
    1. 先关闭 stdout pipe → 让后台日志循环立即退出（不再阻塞）
    2. 本地引用 proc → 防止竞态条件下 training_runtime.training_process 被置为 None
    3. 分阶段杀戮：SIGINT → 优雅 terminate → 暴力 kill
    4. 多次验证进程存活状态
    5. 清理 GPU 显存资源
    """
    with training_runtime.train_state_lock:
        running = training_runtime.training_status['running']
    if not running:
        return {'warning': '没有运行中的训练任务'}

    # 【关键】立即保存本地引用，防止与后台线程产生竞态
    proc = training_runtime.training_process
    old_pid = getattr(proc, 'pid', None) if proc else None

    if training_runtime.remote_training_state.get('active'):
        try:
            cfg = _load_yaml_file(CONFIG_DIR / 'distill_config.yaml') or {}
            train_cfg = dict(cfg.get('training', {}) or {})
            api_cfg = _build_cloud_api_config(train_cfg)
            job_id = str(training_runtime.remote_training_state.get('job_id') or '')
            if job_id:
                _http_json_request('POST', api_cfg['stop_url'], payload={'job_id': job_id}, headers=api_cfg['headers'])
                _update_status_line(f'[REMOTE] 已请求停止云任务: {job_id}')
        except Exception as e:
            _update_status_line(f'[REMOTE] 停止云任务失败: {e}')
        with training_runtime.train_state_lock:
            training_runtime.remote_training_state['active'] = False
            training_runtime.remote_training_state['job_id'] = ''

    if proc and old_pid and old_pid > 0:
        # ═══ Step 1: 先关闭 stdout pipe → 让 _run_training_process_safe 的 for 循环立即退出 ═══
        if proc.stdout and not proc.stdout.closed:
            try:
                proc.stdout.close()
            except Exception:
                pass

        # ═══ Step 2: 发送中断信号（请求优雅退出）═══
        try:
            if os.name == 'nt':
                proc.send_signal(signal.CTRL_C_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception:
            pass

        # ═══ Step 3: 等待进程自行退出（最多 8 秒）═══
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pass

        # ═══ Step 4: 还活着 → 优雅杀进程树 ═══
        if _is_process_alive(old_pid):
            _kill_process_tree(old_pid, force=False)
            time.sleep(1.5)

        # ═══ Step 5: 仍活着 → 暴力杀进程树 ═══
        if _is_process_alive(old_pid):
            _kill_process_tree(old_pid, force=True)
            time.sleep(1)

        # ═══ Step 6: 最终验证 + 告警 ═══
        if _is_process_alive(old_pid):
            warn_msg = f"[STOP_WARN] 进程 PID={old_pid} 停止失败！请手动结束该进程以释放 GPU 显存"
            _update_status_line(warn_msg)

    # ═══ Step 7: 先写入停止日志再清 running，避免 SSE 在 running=False 瞬间关流漏掉本行 ═══
    _update_status_line('训练已被用户停止')

    # ═══ Step 8: 统一更新全局状态（无论是否需要杀进程都执行）═══
    with training_runtime.train_state_lock:
        training_runtime.training_status['running'] = False
        training_runtime.training_status['pid'] = None
    training_runtime.training_process = None

    # ═══ Step 9: 释放锁 + 清理 GPU 显存 ═══
    _release_training_lock()
    _cleanup_gpu_resources()

    return {'status': 'ok', 'message': '训练已停止'}

def get_training_status():
    with training_runtime.train_state_lock:
        snap = {k: (list(v) if k == 'logs' and isinstance(v, list) else v) for k, v in training_runtime.training_status.items()}
    snap['log_count'] = len(snap.get('logs') or [])
    running = bool(snap.get("running"))
    start_time = snap.get("start_time")
    current_epoch = int(snap.get("current_epoch") or 0)
    total_epochs = int(snap.get("total_epochs") or 0)
    elapsed_sec = 0
    expected_sec = 0
    try:
        if running and isinstance(start_time, (int, float)) and start_time > 0:
            elapsed_sec = max(0, int(time.time() - float(start_time)))
            if current_epoch > 0 and total_epochs > 0:
                expected_sec = max(elapsed_sec, int(round((elapsed_sec / current_epoch) * total_epochs)))
    except Exception:
        elapsed_sec = 0
        expected_sec = 0
    snap["elapsed_sec"] = elapsed_sec
    snap["expected_sec"] = expected_sec
    return {'status': 'ok', **snap}

def get_resume_candidates(project: str = Query('runs/distill')):
    project = project or 'runs/distill'
    try:
        project_path = _resolve_project_path(project, allow_external=Path(project).is_absolute())
    except ValueError as e:
        return _error(str(e), 400)
    candidates = _list_resume_candidates(project_path)
    return {
        'status': 'ok',
        'project': str(project_path.relative_to(BASE_DIR)),
        'candidates': candidates
    }

def get_training_logs(offset: int = Query(0), limit: int = Query(100)):
    offset = max(0, int(offset))
    limit = min(5000, max(1, int(limit)))
    with training_runtime.train_state_lock:
        logs = list(training_runtime.training_status['logs'])
    total = len(logs)
    if offset > total:
        offset = total
    return {'status': 'ok', 'logs': logs[offset:offset + limit], 'total': total, 'offset': offset, 'limit': limit}

def download_training_logs():
    """导出当前内存中的训练日志为纯文本（与 /api/train/logs 同源）。"""
    with training_runtime.train_state_lock:
        text = '\n'.join(training_runtime.training_status.get('logs') or [])
    return PlainTextResponse(
        text + ('\n' if text and not text.endswith('\n') else ''),
        headers={'Content-Disposition': 'attachment; filename=training_log.txt'},
    )

def stream_training_logs(offset: int = Query(0)):
    offset = max(0, int(offset))

    def generate():
        # 基于 offset 线性追赶日志，保证重连后不会漏行也不会重复刷整屏。
        next_idx = offset
        while True:
            batch = []
            batch_start = next_idx
            with training_runtime.train_state_lock:
                buf = training_runtime.training_status['logs']
                n = len(buf)
                running = training_runtime.training_status['running']
                if next_idx < n:
                    batch = list(buf[next_idx:n])
                    batch_start = next_idx
                    next_idx = n
                elif running:
                    training_runtime.train_log_cond.wait(timeout=2.0)
                else:
                    break

            if batch:
                for idx, line in enumerate(batch, start=batch_start + 1):
                    payload = {'line': line, 'idx': idx}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                continue

            with training_runtime.train_state_lock:
                is_done = (not training_runtime.training_status['running']) and next_idx >= len(training_runtime.training_status['logs'])
            if is_done:
                break

            if not batch:
                yield ': keepalive\n\n'
        yield 'event: done\ndata: {}\n\n'

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'}
    )
