from __future__ import annotations

import copy
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from web.core.paths import BASE_DIR, CONFIG_DIR
from web.services import training_runtime
from web.services.backend_common import _candidate_output_roots, _error, _list_resume_candidates, _load_yaml_file, _normalize_compute_provider, _resolve_project_path

_TRAIN_LOCK_FILE = BASE_DIR / '.training.lock'

def _acquire_os_file_lock(lock_path: Path, timeout: float = 0) -> bool:
    """
    获取 OS 级别独占文件锁。
    
    与旧版 _TrainingLock 的本质区别：
      旧版用 write_text+rename 模拟锁 → 存在 TOCTOU 竞态窗口
      此版本使用 fcntl.flock / msvcrt.locking → 内核保证原子性
    
    Args:
        lock_path: 锁文件路径
        timeout: 超时秒数，0=非阻塞立即返回
    
    Returns:
        True=获取成功，False=被其他进程持有
    """
    try:
        import msvcrt
        fd = open(str(lock_path), 'w')
        # Windows: LOCK_EX | LOCK_NB（非阻塞排他锁）
        msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
        training_runtime.train_fd = fd
        fd.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
        fd.flush()
        return True
    except ImportError:
        pass
    except (OSError, IOError):
        return False

    # Linux/macOS: fcntl.flock
    try:
        import fcntl
        fd = open(str(lock_path), 'w')
        flags = fcntl.LOCK_EX | fcntl.LOCK_NB  # 排他 + 非阻塞
        fcntl.flock(fd.fileno(), flags)
        training_runtime.train_fd = fd
        fd.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
        fd.flush()
        return True
    except (OSError, IOError):
        return False

def _release_os_file_lock():
    """释放 OS 文件锁"""
    if training_runtime.train_fd is not None:
        try:
            import msvcrt
            msvcrt.unlocking(training_runtime.train_fd.fileno(), 1)
        except ImportError:
            pass
        except Exception:
            pass
        try:
            import fcntl
            fcntl.flock(training_runtime.train_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            training_runtime.train_fd.close()
        except Exception:
            pass
        training_runtime.train_fd = None

def _acquire_training_lock(timeout: float = 0) -> bool:
    """获取训练互斥锁的统一入口"""
    return _acquire_os_file_lock(_TRAIN_LOCK_FILE, timeout)

def _release_training_lock():
    """释放训练互斥锁的统一入口"""
    _release_os_file_lock()

def _is_process_alive(pid: int) -> bool:
    """跨平台进程存活检测"""
    if pid <= 0:
        return False
    try:
        import psutil as _psutil
        return _psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

def _kill_process_tree(pid: int, force: bool = False) -> int:
    """
    杀死整个进程树（主进程 + 所有子/孙进程）。
    
    【关键改进】旧版 _kill_old_training 只杀直接子进程 Popen 对象，
    实际上 ultralytics 训练运行在孙子进程中（Popen→python→ultralytics），
    导致孙子进程逃逸 → GPU 显存不释放 → 新旧进程同时占 GPU → OOM。
    
    Returns:
        成功终止的进程数量
    """
    killed_count = 0
    try:
        import psutil
    except ImportError:
        # 无 psutil 时降级为基础 kill
        try:
            os.kill(pid, signal.SIGTERM if not force else signal.SIGKILL)
            return 1
        except Exception:
            return 0

    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # 先杀所有子进程
        for child in children:
            try:
                child.terminate()
                killed_count += 1
            except psutil.NoSuchProcess:
                pass
        
        # 等待子进程退出（最多 3 秒）
        gone, alive = psutil.wait_procs(children, timeout=3)
        
        # 还活着的暴力杀
        for p in alive:
            try:
                p.kill()
                killed_count += 1
            except psutil.NoSuchProcess:
                pass
        
        # 最后杀主进程
        try:
            parent.terminate()
            killed_count += 1
        except psutil.NoSuchProcess:
            pass
        
        try:
            parent.wait(timeout=3)
        except psutil.TimeoutExpired:
            try:
                parent.kill()
            except Exception:
                pass
        
        return killed_count
    except psutil.NoSuchProcess:
        return 0
    except Exception:
        return 0

def _scan_and_kill_stale_training_processes(exclude_pid: int = None) -> dict:
    """
    扫描并杀死残留的训练进程。
    
    场景：Web 服务重启后内存中的 training_runtime.training_process 引用丢失，
    但旧的 python.exe 训练进程仍在占用 GPU。
    通过扫描 .training.lock 文件和进程命令行特征来发现并清理。
    
    Returns:
        {'found': int, 'killed': int, 'details': str}
    """
    result = {'found': 0, 'killed': 0, 'details': ''}
    stale_pids = []
    
    # 1. 从锁文件读取 PID
    if _TRAIN_LOCK_FILE.exists():
        try:
            content = _TRAIN_LOCK_FILE.read_text().strip().split('\n')
            old_pid = int(content[0]) if content else -1
            if old_pid > 0 and old_pid != exclude_pid and _is_process_alive(old_pid):
                stale_pids.append(old_pid)
        except (ValueError, IndexError, OSError):
            pass
    
    # 2. 扫描包含训练特征的 python 进程（补充检测）
    try:
        import psutil
        our_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'cmdline', 'name']):
            try:
                if proc.pid == our_pid or proc.pid == exclude_pid:
                    continue
                if proc.info['name'] != 'python.exe' and proc.info['name'] != 'python':
                    continue
                cmdline = proc.info['cmdline'] or []
                cmdline_str = ' '.join(cmdline).lower()
                # 匹配训练脚本的特征命令行
                if any(kw in cmdline_str for kw in [
                    'train_with_distill', 'main.py train', 
                    'yolo train', 'ultralytics',
                    '--resume'
                ]):
                    if proc.pid not in stale_pids:
                        stale_pids.append(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        pass
    except Exception:
        pass
    
    result['found'] = len(stale_pids)
    
    # 3. 杀死所有发现的残留进程
    for spid in stale_pids:
        k = _kill_process_tree(spid, force=True)
        result['killed'] += k
        result['details'] += f"PID={spid}(killed={k}); "
    
    # 4. 删除僵尸锁文件
    if stale_pids:
        try:
            _TRAIN_LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    
    return result

def _strip_ansi(text: str) -> str:
    """去掉 ESC 序列（ultralytics 自带 TQDM 使用 \\r + \\033[K，原样进管道会触发误过滤）。"""
    if not text:
        return text
    return re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)

def _extract_epoch_progress(line: str):
    """从不同训练日志格式中提取 (current_epoch, total_epochs)。"""
    if not line:
        return None

    patterns = [
        # 结构化日志: [EPOCH_START] epoch=1 total=10 / [EPOCH_PROGRESS] epoch=1 total=10
        r"\bepoch\s*=\s*(\d+)\s+total\s*=\s*(\d+)\b",
        # 常见格式: Epoch 1/10 或 Epoch: 1 / 10
        r"\bEpoch\s*[:=]?\s*(\d+)\s*/\s*(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            if total > 0 and 0 <= current <= total:
                return current, total

    # 兼容 YOLO 训练进度行（包含显存列）:
    # " 1/10  2.98G  1.266 1.555 ... 640: 12% ... 1/8 1.7s/it"
    # 注意：实时行通常不含 "GPU_mem/box_loss" 字面表头，因此不再依赖这些关键词。
    yolo_row = re.search(r"^\s*(\d+)\s*/\s*(\d+)\s+\d+(?:\.\d+)?G\b", line)
    if yolo_row:
        current = int(yolo_row.group(1))
        total = int(yolo_row.group(2))
        if total > 0 and 0 <= current <= total:
            return current, total

    # 兜底：保留历史裸格式 + 关键词判定
    bare = re.search(r"^\s*(\d+)\s*/\s*(\d+)\b", line)
    if bare and re.search(r"\b(GPU_mem|box_loss|cls_loss|dfl_loss|Instances|Size|it/s|s/it)\b", line, re.IGNORECASE):
        current = int(bare.group(1))
        total = int(bare.group(2))
        if total > 0 and 0 <= current <= total:
            return current, total

    return None

def _update_status_line(line: str):
    """原样转发 Ultralytics / 训练子进程输出（仅剥 ANSI、去首尾空白）；不做语义过滤与去重。"""
    if not line:
        return
    clean_line = _strip_ansi(line).rstrip('\r\n')
    if not clean_line.strip():
        return

    with training_runtime.train_state_lock:
        training_runtime.training_status['logs'].append(clean_line)
        training_runtime.train_log_cond.notify_all()
        # 仅在训练未运行时裁剪：训练中裁剪会打乱长连接按索引追赶 logs 的语义，导致漏推尾部日志
        if len(training_runtime.training_status['logs']) > 8000 and not training_runtime.training_status['running']:
            training_runtime.training_status['logs'] = training_runtime.training_status['logs'][-4000:]

        progress = _extract_epoch_progress(clean_line)
        if progress:
            training_runtime.training_status['current_epoch'], training_runtime.training_status['total_epochs'] = progress

def _http_json_request(method: str, url: str, payload: dict | None = None, headers: dict | None = None, timeout: float = 20.0):
    req_headers = {'Content-Type': 'application/json'}
    if isinstance(headers, dict):
        req_headers.update({str(k): str(v) for k, v in headers.items() if k})
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url=url, data=data, headers=req_headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode('utf-8', errors='replace')
        return json.loads(body) if body else {}

def _set_auth_headers(headers: dict | None, api_key: str | None, prefer_x_api_key: bool = False) -> dict:
    out = dict(headers or {})
    key = str(api_key or '').strip()
    if not key:
        return out
    lower_keys = {str(k).lower() for k in out.keys()}
    bearer_value = key if re.match(r'^\s*bearer\s+', key, flags=re.IGNORECASE) else f'Bearer {key}'
    if 'authorization' not in lower_keys:
        out['Authorization'] = bearer_value
    if prefer_x_api_key and 'x-api-key' not in lower_keys:
        out['x-api-key'] = key
    return out

def _build_cloud_api_config(train_cfg: dict):
    cloud_api = dict(train_cfg.get('cloud_api', {}) or {})
    base_url = str(cloud_api.get('base_url', '') or '').strip().rstrip('/')
    if not base_url:
        raise ValueError('云训练 API 缺少 base_url')
    submit_path = str(cloud_api.get('submit_path', '/train/start') or '/train/start').strip()
    status_path = str(cloud_api.get('status_path', '/train/status') or '/train/status').strip()
    logs_path = str(cloud_api.get('logs_path', '/train/logs') or '/train/logs').strip()
    stop_path = str(cloud_api.get('stop_path', '/train/stop') or '/train/stop').strip()
    auth_token = str(cloud_api.get('token', '') or '').strip()
    headers = dict(cloud_api.get('headers', {}) or {})
    if auth_token and 'Authorization' not in headers:
        headers['Authorization'] = auth_token
    return {
        'base_url': base_url,
        'submit_url': f"{base_url}{submit_path}",
        'status_url': f"{base_url}{status_path}",
        'logs_url': f"{base_url}{logs_path}",
        'stop_url': f"{base_url}{stop_path}",
        'headers': headers,
        'poll_interval_sec': float(cloud_api.get('poll_interval_sec', 3)),
    }

def _resolve_dataset_via_api(train_cfg: dict, cfg: dict, mode: str, checkpoint: str | None):
    dataset_api = dict(train_cfg.get('dataset_api', {}) or {})
    source = str(dataset_api.get('source', '') or '').strip().lower()
    enabled = bool(dataset_api.get('enabled', False) or source == 'api')
    resolve_url = str(dataset_api.get('resolve_url', '') or '').strip()
    if not enabled or not resolve_url:
        return None

    headers = dict(dataset_api.get('headers', {}) or {})
    token = str(dataset_api.get('token', '') or '').strip()
    if not token:
        token = str((train_cfg.get('cloud_api') or {}).get('token', '') or '').strip()
    if token and 'Authorization' not in headers:
        headers['Authorization'] = token

    request_payload = dataset_api.get('request_body')
    if not isinstance(request_payload, dict):
        request_payload = {
            'dataset_name': dataset_api.get('dataset_name', ''),
            'config': cfg,
            'mode': mode,
            'checkpoint': checkpoint,
        }
    timeout_sec = float(dataset_api.get('timeout_sec', 30))
    result = _http_json_request('POST', resolve_url, payload=request_payload, headers=headers, timeout=timeout_sec)
    if not isinstance(result, dict):
        raise ValueError('数据集 API 返回格式非法（需为 JSON 对象）')

    data_yaml = str(
        result.get('data_yaml')
        or result.get('dataset_yaml')
        or result.get('dataset_path')
        or ''
    ).strip()
    dataset_id = str(result.get('dataset_id') or result.get('id') or '').strip()
    if not data_yaml and not dataset_id:
        raise ValueError('数据集 API 未返回 data_yaml 或 dataset_id')

    out = dict(result)
    out['data_yaml'] = data_yaml
    out['dataset_id'] = dataset_id
    return out

def _remote_polling_loop(api_cfg: dict, job_id: str):
    while True:
        with training_runtime.train_state_lock:
            if not training_runtime.remote_training_state.get('active'):
                break
            logs_offset = int(training_runtime.remote_training_state.get('logs_offset', 0) or 0)

        try:
            status_qs = urllib.parse.urlencode({'job_id': job_id})
            status_data = _http_json_request('GET', f"{api_cfg['status_url']}?{status_qs}", headers=api_cfg['headers'])
            state = str(status_data.get('state', '') or '').lower()
            current_epoch = int(status_data.get('current_epoch', 0) or 0)
            total_epochs = int(status_data.get('total_epochs', 0) or 0)
            with training_runtime.train_state_lock:
                training_runtime.training_status['current_epoch'] = current_epoch
                training_runtime.training_status['total_epochs'] = total_epochs

            logs_qs = urllib.parse.urlencode({'job_id': job_id, 'offset': logs_offset, 'limit': 200})
            logs_data = _http_json_request('GET', f"{api_cfg['logs_url']}?{logs_qs}", headers=api_cfg['headers'])
            lines = list(logs_data.get('logs') or [])
            if lines:
                with training_runtime.train_state_lock:
                    for line in lines:
                        _update_status_line(str(line))
                    training_runtime.remote_training_state['logs_offset'] = logs_offset + len(lines)

            if state in {'completed', 'failed', 'stopped', 'cancelled', 'done', 'success'}:
                with training_runtime.train_state_lock:
                    training_runtime.training_status['running'] = False
                    training_runtime.training_status['pid'] = None
                    training_runtime.remote_training_state['active'] = False
                _update_status_line(f"[REMOTE] 云训练结束，状态: {state or 'unknown'}")
                break
        except Exception as e:
            _update_status_line(f"[REMOTE] 轮询异常: {e}")

        time.sleep(max(1.0, float(api_cfg.get('poll_interval_sec', 3.0))))

def _iter_pipe_lines(stdout_pipe, chunk_size: int = 4096):
    """按块读取 stdout，并将 \\r/\\n 统一视为行结束，保证 tqdm 刷新也能实时输出。"""
    if stdout_pipe is None:
        return
    pending = ''
    while True:
        raw = stdout_pipe.read(chunk_size)
        if not raw:
            break
        text = raw.decode('utf-8', errors='replace') if isinstance(raw, (bytes, bytearray)) else str(raw)
        if not text:
            continue
        normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        pending += normalized
        while '\n' in pending:
            line, pending = pending.split('\n', 1)
            yield line
    if pending:
        yield pending

def _kill_old_training():
    """
    强制终止旧训练进程树并等待完全退出 — 彻底版
    
    【关键改进】使用 _kill_process_tree 杀整个进程树，不再只杀 Popen 直接子进程。
    解决：Popen 启动的 python.exe → ultralytics 子进程逃逸 → GPU 显存不释放
    """
    if training_runtime.training_process is None:
        return
    old_pid = getattr(training_runtime.training_process, 'pid', None)

    killed = 0
    try:
        if old_pid and old_pid > 0:
            # 先尝试优雅终止
            try:
                if os.name == 'nt':
                    training_runtime.training_process.send_signal(signal.CTRL_C_EVENT)
                else:
                    training_runtime.training_process.send_signal(signal.SIGINT)
            except Exception:
                pass

            # 等待正常退出
            try:
                training_runtime.training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

            # 还活着就暴力杀进程树
            if training_runtime.training_process.poll() is None:
                killed = _kill_process_tree(old_pid, force=False)

                # 等一下看是否都死了
                time.sleep(1)
                if _is_process_alive(old_pid):
                    _kill_process_tree(old_pid, force=True)
                    time.sleep(0.5)

            # 关闭 stdout pipe
            if training_runtime.training_process.stdout and not training_runtime.training_process.stdout.closed:
                try:
                    training_runtime.training_process.stdout.close()
                except Exception:
                    pass

        msg = f"旧训练进程已终止 (PID={old_pid}, 树中杀死={killed}个)"
        _update_status_line(msg)
    finally:
        with training_runtime.train_state_lock:
            training_runtime.training_status['running'] = False
            training_runtime.training_status['pid'] = None
        training_runtime.training_process = None

def _run_training_process_safe(cmd):
    """
    安全版训练进程启动 — 在锁已持有的前提下执行
    
    此函数的调用者 (start_training) 必须已经：
      1. 持有 training_runtime.train_thread_lock
      2. 已获取 _TRAIN_LOCK_FILE 的 OS 文件锁
      3. 已杀掉所有旧训练进程
      4. 已清理 GPU 资源
      
    本函数只负责：启动新子进程 → 实时内存监控 → 读取日志 → 进程结束 → 清理状态
    【新增】内置内存超限检测：一旦子进程树总内存超过阈值立即杀掉整棵进程树
    """

    # 内存监控配置（单位：字节）
    # 默认限制 12GB — 超过此值视为异常（正常蒸馏训练约 4-8GB）
    MAX_MEMORY_BYTES = int(os.environ.get('EDGE_TRAIN_MAX_MEM_GB', '12')) * 1024 * 1024 * 1024
    # 监控间隔：每隔多少行日志检查一次内存（平衡性能与响应速度）
    MEMORY_CHECK_INTERVAL = 50
    # 连续超限次数阈值（防止瞬时峰值误杀）
    MEMORY_OVERLIMIT_THRESHOLD = 3

    proc = None
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        # 子进程无 TTY 时强制行缓冲，避免 ultralytics / logging 长时间不刷到管道
        env['PYTHONUNBUFFERED'] = '1'
        # 训练子进程内将 TQDM 的 \\r 刷新改为按行输出，否则管道 readline 只能收到表头收不到每 batch 数值行
        env['EDGE_WEB_LOG'] = '1'

        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        preexec_fn = None if os.name == 'nt' else os.setsid

        # Windows 上 text=True + bufsize=1 对管道往往仍块缓冲，改为二进制 bufsize=0 + readline，尽快收到每一行
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=env,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        training_runtime.training_process = proc
        with training_runtime.train_state_lock:
            training_runtime.training_status['pid'] = proc.pid

        _update_status_line(f'[MEM_GUARD] 内存监控已启用 | 上限={MAX_MEMORY_BYTES // (1024**3)}GB | '
                           f'检查间隔={MEMORY_CHECK_INTERVAL}行 | 连续超限={MEMORY_OVERLIMIT_THRESHOLD}次')

        line_count = 0

        # 全程使用局部 proc，避免 stop_training 把全局 training_runtime.training_process 置 None 后出现 .wait() 竞态
        for raw_line in _iter_pipe_lines(proc.stdout):
            line_count += 1
            _update_status_line(raw_line)

            # ════════════════════════════════════════════
            # 定期内存安全检查
            # ════════════════════════════════════════════
            if line_count % MEMORY_CHECK_INTERVAL != 0:
                continue

            if not _check_and_enforce_memory_limit(
                pid=proc.pid,
                max_bytes=MAX_MEMORY_BYTES,
                threshold=MEMORY_OVERLIMIT_THRESHOLD,
            ):
                # _check_and_enforce_memory_limit 返回 False 表示已触发杀戮
                # 此时进程已被杀死或正在被杀，退出循环
                break

        try:
            if proc.stdout and not proc.stdout.closed:
                proc.stdout.close()
        except Exception:
            pass

        # 等待进程完全退出（如果还没退的话）
        if proc.poll() is None:
            proc.wait()
            
    except Exception as e:
        _update_status_line(f"训练异常: {e}")
    finally:
        # 清理状态 + 释放文件锁
        with training_runtime.train_state_lock:
            training_runtime.training_status['running'] = False
            training_runtime.training_status['pid'] = None
        training_runtime.training_process = None
        _release_training_lock()

_mem_overlimit_count = 0

def _check_and_enforce_memory_limit(pid: int, max_bytes: int, threshold: int) -> bool:
    """
    检查指定 PID 及其子进程的总内存占用。
    
    Returns:
        True  = 内存正常，继续运行
        False = 已触发强制杀戮，调用者应停止读取日志循环
    """
    global _mem_overlimit_count
    
    total_rss = 0
    process_list = []
    
    try:
        import psutil
        
        try:
            parent = psutil.Process(pid)
            process_list.append(parent)
            process_list.extend(parent.children(recursive=True))
        except psutil.NoSuchProcess:
            return True  # 进程已死，不算异常
        
        # 累加所有进程的 RSS（常驻物理内存）
        for proc in process_list:
            try:
                mem_info = proc.memory_info()
                total_rss += mem_info.rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
    except ImportError:
        return True  # 无 psutil 时跳过监控
    
    except Exception:
        return True  # 其他异常跳过本次检查
    
    # 判断是否超限
    mb_used = total_rss / (1024 * 1024)
    mb_limit = max_bytes / (1024 * 1024)
    
    if total_rss > max_bytes:
        _mem_overlimit_count += 1
        
        if _mem_overlimit_count >= threshold:
            # 连续超限达到阈值 → 强制杀掉整棵进程树
            msg = (f"[MEM_ALERT] ⚠️ 内存严重超标！当前 {mb_used:.1f}MB > 限制 {mb_limit:.0f}MB "
                   f"(连续{_mem_overlimit_count}次)，强制终止训练进程树...")
            _update_status_line(msg)
            killed = _kill_process_tree(pid, force=True)
            time.sleep(1)  # 等待系统回收
            _update_status_line(f"[MEM_ALERT] 已强制终止 {killed} 个进程，释放 {mb_used:.0f}MB 内存")
            return False
        else:
            # 未达阈值但已超限 → 警告
            _update_status_line(
                f"[MEM_WARN] 内存偏高: {mb_used:.0f}MB / {mb_limit:.0f}MB "
                f"({_mem_overlimit_count}/{threshold})，继续观察..."
            )
            return True
    else:
        # 内存正常 → 重置计数器
        if _mem_overlimit_count > 0:
            _mem_overlimit_count = 0
        return True

def _cleanup_gpu_resources():
    """清理残留 GPU 显存资源"""
    try:
        import gc as _gc2
        import torch as _torch2
        _gc2.collect()
        if _torch2.cuda.is_available():
            _torch2.cuda.empty_cache()
            _torch2.cuda.reset_peak_memory_stats()
            if hasattr(_torch2.cuda, 'synchronize'):
                try:
                    _torch2.cuda.synchronize()
                except Exception:
                    pass
    except ImportError:
        pass
    except Exception:
        pass

def _wait_for_gpu_free(timeout_sec: float = 15.0) -> bool:
    """
    等待 GPU 显存释放到安全水平。
    
    通过 nvidia-smi 或 torch 检测 GPU 使用情况，
    在启动新训练前确保旧进程的显存已被回收。
    
    Returns:
        True=GPU 已就绪可使用，False=超时
    """
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            import torch as _t
            if _t.cuda.is_available():
                try:
                    test_tensor = _t.zeros(1, device='cuda')
                    del test_tensor
                    _t.cuda.empty_cache()
                    return True
                except RuntimeError as _e:
                    if 'out of memory' in str(_e).lower():
                        _update_status_line(f'[GPU] 等待显存释放中... ({int(deadline - time.time())}s)')
                        time.sleep(2)
                        continue
            return True
        except ImportError:
            return True
        except Exception:
            time.sleep(2)
    return False
