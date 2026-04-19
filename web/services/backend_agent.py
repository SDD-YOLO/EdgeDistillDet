from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi.responses import StreamingResponse

from web.core.paths import get_agent_history_dir, get_config_dir
from web.schemas import AgentModelInvokeRequest, AgentPatchApplyRequest, AgentPatchPreviewRequest, AgentPatchValidateRequest, AgentRunHistoryRollbackRequest, AgentToolExecuteRequest
from web.services import backend_state
from web.services.backend_common import _error, _load_yaml_file, _save_yaml_file
from web.services import training_runtime
from web.services.backend_metrics import get_metrics
from web.services.backend_train_runtime import _http_json_request, _set_auth_headers

AGENT_TOOL_CONTRACT_VERSION = 'v1'

_AGENT_PATCH_TTL = 600.0

_agent_patch_store = {}

_AGENT_ALLOWED_TOP_LEVEL = frozenset({'distillation', 'training', 'output'})
_AGENT_PATCH_TOP_ORDER = ('distillation', 'training', 'output')

_AGENT_MAX_ROLLBACKS_PER_RUN = 5

def _safe_run_id(run_id: str) -> str:
    rid = str(run_id or 'default').strip()
    rid = re.sub(r'[^a-zA-Z0-9._-]+', '_', rid)
    return rid[:128] or 'default'

def _agent_history_file(run_id: str) -> Path:
    hdir = get_agent_history_dir()
    hdir.mkdir(parents=True, exist_ok=True)
    return hdir / f"{_safe_run_id(run_id)}.json"

def _agent_load_history(run_id: str) -> list[dict]:
    path = _agent_history_file(run_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
    except Exception:
        pass
    return []

def _agent_save_history(run_id: str, entries: list[dict]) -> None:
    path = _agent_history_file(run_id)
    keep = list(entries or [])[-_AGENT_MAX_ROLLBACKS_PER_RUN:]
    path.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding='utf-8')

def _agent_request_hash(run_id: str, patch: dict) -> str:
    raw = json.dumps({'run_id': _safe_run_id(run_id), 'patch': patch}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def _prune_agent_patch_store():
    now = time.time()
    for k, rec in list(_agent_patch_store.items()):
        if rec.get('expires', 0) < now:
            _agent_patch_store.pop(k, None)

def _deep_merge_shallow(dst: dict, src: dict) -> dict:
    out = copy.deepcopy(dst) if isinstance(dst, dict) else {}
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_shallow(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def _merge_distill_patch(base: dict, patch: dict) -> dict:
    if not isinstance(patch, dict) or not patch:
        raise ValueError('patch 必须为非空对象')
    extra = set(patch.keys()) - _AGENT_ALLOWED_TOP_LEVEL
    if extra:
        raise ValueError('不允许的顶层键: ' + ', '.join(sorted(extra)))
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    for top in patch:
        sub = patch[top]
        if not isinstance(sub, dict):
            merged[top] = copy.deepcopy(sub)
            continue
        cur = merged.get(top)
        merged[top] = _deep_merge_shallow(cur if isinstance(cur, dict) else {}, sub)
    return merged


def _serialize_diff_value(val: Any) -> Any:
    """JSON 友好、适合前端展示的叶子值。"""
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    try:
        json.dumps(val, ensure_ascii=False)
        return val
    except (TypeError, ValueError):
        return repr(val)


def _leaf_config_diff(base: dict | None, merged: dict | None) -> dict[str, Any]:
    """
    对比合并前后的 distill 配置，产出叶子级路径变更（仅 distillation / training / output）。
    path 使用点号连接，如 training.lr0。
    """
    rows: list[dict[str, Any]] = []

    def walk(b: Any, m: Any, path: str) -> None:
        if isinstance(b, dict) and isinstance(m, dict):
            keys = set(b.keys()) | set(m.keys())
            for k in sorted(keys, key=lambda x: str(x)):
                sub_path = f'{path}.{k}' if path else str(k)
                if k not in b:
                    walk(None, m[k], sub_path)
                elif k not in m:
                    walk(b[k], None, sub_path)
                else:
                    walk(b[k], m[k], sub_path)
            return
        if b is None and isinstance(m, dict):
            for k in sorted(m.keys(), key=lambda x: str(x)):
                sub_path = f'{path}.{k}' if path else str(k)
                walk(None, m[k], sub_path)
            return
        if m is None and isinstance(b, dict):
            for k in sorted(b.keys(), key=lambda x: str(x)):
                sub_path = f'{path}.{k}' if path else str(k)
                walk(b[k], None, sub_path)
            return
        if b == m:
            return
        if b is None:
            kind = 'added'
        elif m is None:
            kind = 'removed'
        else:
            kind = 'changed'
        rows.append({
            'path': path or '(root)',
            'kind': kind,
            'before': _serialize_diff_value(b),
            'after': _serialize_diff_value(m),
        })

    b0 = base if isinstance(base, dict) else {}
    m0 = merged if isinstance(merged, dict) else {}
    for top in sorted(_AGENT_ALLOWED_TOP_LEVEL):
        if top not in b0 and top not in m0:
            continue
        if top not in b0:
            walk(None, m0[top], top)
        elif top not in m0:
            walk(b0[top], None, top)
        else:
            walk(b0[top], m0[top], top)

    return {
        'paths': rows,
        'stats': {
            'changed': len(rows),
        },
    }


def _collect_leaf_paths_from_patch(patch: dict | None) -> set[str]:
    """Patch 中显式出现的叶子路径（点号），用于审批摘要仅反映 Agent 声明要改动的键。"""
    out: set[str] = set()

    def walk(obj: Any, prefix: str) -> None:
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            sub_path = f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v, dict) and v:
                walk(v, sub_path)
            else:
                out.add(sub_path)

    if not isinstance(patch, dict):
        return out
    for top in _AGENT_PATCH_TOP_ORDER:
        if top not in patch:
            continue
        sub = patch[top]
        if isinstance(sub, dict):
            walk(sub, top)
        else:
            out.add(top)
    return out


def _filter_change_summary_to_patch_declared(change_summary: dict, patch: dict) -> dict:
    """仅保留 patch 叶子路径上的 diff 行，避免审批表列出未在 patch 中声明的字段（含类型漂移导致的伪差异）。"""
    allowed = _collect_leaf_paths_from_patch(patch)
    paths = [r for r in (change_summary.get('paths') or []) if isinstance(r, dict) and r.get('path') in allowed]
    return {
        'paths': paths,
        'stats': {'changed': len(paths)},
    }


def _agent_record_history(run_id: str, before_cfg: dict, after_cfg: dict, operator: str, reason: str, action: str) -> dict:
    hist = _agent_load_history(run_id)
    version = int(hist[-1]['version']) + 1 if hist else 1
    rec = {
        'version': version,
        'run_id': _safe_run_id(run_id),
        'timestamp': datetime.now().isoformat(),
        'operator': str(operator or 'user'),
        'reason': str(reason or ''),
        'action': str(action or 'apply'),
        'before_config': before_cfg,
        'after_config': after_cfg,
    }
    hist.append(rec)
    _agent_save_history(run_id, hist)
    return rec

def _agent_validate_patch(patch: dict, strict: bool = True) -> dict:
    errors = []
    warnings = []
    if not isinstance(patch, dict) or not patch:
        errors.append('patch 必须是非空对象')
        return {'valid': False, 'errors': errors, 'warnings': warnings}
    extra = set(patch.keys()) - _AGENT_ALLOWED_TOP_LEVEL
    if extra:
        errors.append('不允许的顶层键: ' + ', '.join(sorted(extra)))
    t = patch.get('training')
    if isinstance(t, dict):
        if 'epochs' in t:
            try:
                if int(t.get('epochs')) <= 0:
                    errors.append('training.epochs 必须大于 0')
            except Exception:
                errors.append('training.epochs 必须为整数')
        if 'batch' in t:
            try:
                b = int(t.get('batch'))
                if b == 0:
                    errors.append('training.batch 不能为 0')
            except Exception:
                errors.append('training.batch 必须为整数')
        if 'lr0' in t:
            try:
                lr0 = float(t.get('lr0'))
                if lr0 <= 0:
                    errors.append('training.lr0 必须大于 0')
                if lr0 > 1:
                    warnings.append('training.lr0 偏大，可能导致训练不稳定')
            except Exception:
                errors.append('training.lr0 必须为数值')
    d = patch.get('distillation')
    if isinstance(d, dict):
        for k in ('alpha_init', 'w_kd', 'w_focal', 'w_feat'):
            if k in d:
                try:
                    v = float(d.get(k))
                    if v < 0:
                        errors.append(f'distillation.{k} 不能小于 0')
                except Exception:
                    errors.append(f'distillation.{k} 必须为数值')
        if 'T_min' in d and 'T_max' in d:
            try:
                if float(d.get('T_min')) > float(d.get('T_max')):
                    errors.append('distillation.T_min 不能大于 T_max')
            except Exception:
                errors.append('distillation.T_min/T_max 必须为数值')
    if strict and errors:
        return {'valid': False, 'errors': errors, 'warnings': warnings}
    return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}

def _agent_training_is_active() -> bool:
    with training_runtime.train_state_lock:
        if training_runtime.training_status.get('running'):
            return True
        if (training_runtime.remote_training_state or {}).get('active'):
            return True
    return False


def _agent_metrics_summary_for_tools() -> dict:
    """训练进行中不向 Agent 暴露未定型指标；仅在空闲时附带 metrics 摘要。"""
    if _agent_training_is_active():
        return {
            'status': 'training_in_progress',
            'hint': '训练尚未结束，指标未定型；请勿将本快照当作最终训练结果，待训练完成后再分析。',
        }
    metrics: dict = {}
    try:
        m = get_metrics()
        if isinstance(m, dict):
            metrics = {
                'csv_count': len(m.get('csv_metrics') or []),
                'source': m.get('source') or '',
                'overview_stats': m.get('overview_stats') or {},
            }
    except Exception:
        metrics = {}
    return metrics


def _agent_get_context(run_id: str, config_name: str = 'distill_config.yaml') -> dict:
    config_path = get_config_dir() / config_name
    cfg = _load_yaml_file(config_path) or {}
    return {
        'run_id': _safe_run_id(run_id),
        'config_file': config_name,
        'allowed_top_level': sorted(_AGENT_ALLOWED_TOP_LEVEL),
        'current_config': cfg,
        'metrics_summary': _agent_metrics_summary_for_tools(),
    }

def _agent_analyze_params(run_id: str, objective: str, config_name: str = 'distill_config.yaml') -> dict:
    ctx = _agent_get_context(run_id=run_id, config_name=config_name)
    if _agent_training_is_active():
        return {
            'run_id': _safe_run_id(run_id),
            'objective': objective,
            'analysis': [{'focus': '等待训练结束', 'suggestion': '当前有训练任务进行中，指标未定型；请在训练完成后再调用本工具或基于最终 results.csv 分析。', 'risk': ''}],
            'need_approval': False,
            'training_in_progress': True,
            'context': ctx,
        }
    advice = []
    low_obj = str(objective or '').lower()
    if any(x in low_obj for x in ('速度', 'latency', '时延')):
        advice.append({'focus': '降低计算开销', 'suggestion': '优先减小 imgsz/batch 或使用更轻 student_weight', 'risk': '精度可能下降'})
    if any(x in low_obj for x in ('精度', 'accuracy', 'map')):
        advice.append({'focus': '提升检测精度', 'suggestion': '可适当提高 epochs 并调整 w_kd/w_focal', 'risk': '训练时长增加'})
    if any(x in low_obj for x in ('显存', 'memory', 'oom')):
        advice.append({'focus': '控制显存', 'suggestion': '减小 batch，开启 amp，降低 imgsz', 'risk': '收敛速度可能变慢'})
    if not advice:
        advice.append({'focus': '通用优化', 'suggestion': '先小步调整 alpha_init、w_kd、lr0 并观察曲线', 'risk': '需多轮试验'})
    return {'run_id': _safe_run_id(run_id), 'objective': objective, 'analysis': advice, 'need_approval': True, 'context': ctx}


def _bump_declared_leaf(v: Any) -> Any | None:
    """对「补丁声明值与磁盘相同」的叶子做极小变更，便于产生可审批 diff；无法处理则返回 None。"""
    if isinstance(v, bool):
        return not v
    if isinstance(v, int) and not isinstance(v, bool):
        return int(v) + 1
    if isinstance(v, float):
        return float(v) + 1e-6 if v != 0.0 else 1e-6
    return None


def _single_leaf_epsilon_declared_patch(base: dict, declared: dict) -> dict:
    """
    当声明的 patch 与磁盘合并后无差异时，仅在 **declared 中已出现的叶子路径** 上改一个字段，
    不引入任何未在 declared 中出现的键（避免审批 kd 却写入 temperature 等）。
    遍历顺序：distillation → training → output；段内键名按字典序，先命中先 bump。
    """
    if not isinstance(declared, dict) or not declared:
        return {}
    for top in _AGENT_PATCH_TOP_ORDER:
        if top not in declared or top not in _AGENT_ALLOWED_TOP_LEVEL:
            continue
        sub = declared[top]
        if not isinstance(sub, dict):
            cur = base.get(top)
            if cur == sub:
                b = _bump_declared_leaf(sub)
                if b is not None:
                    return {top: b}
            continue
        base_sub = base.get(top) if isinstance(base.get(top), dict) else {}
        for k in sorted(sub.keys()):
            v = sub[k]
            if isinstance(v, dict):
                base_inner = base_sub.get(k) if isinstance(base_sub.get(k), dict) else {}
                if not isinstance(base_inner, dict):
                    base_inner = {}
                for k2 in sorted(v.keys()):
                    v2 = v[k2]
                    if isinstance(v2, dict):
                        continue
                    cur2 = base_inner.get(k2) if isinstance(base_inner, dict) else None
                    if cur2 != v2:
                        continue
                    b2 = _bump_declared_leaf(v2)
                    if b2 is not None:
                        return {top: {k: {k2: b2}}}
                continue
            cur = base_sub.get(k) if isinstance(base_sub, dict) else None
            if cur != v:
                continue
            b = _bump_declared_leaf(v)
            if b is not None:
                return {top: {k: b}}
    return {}


def _agent_propose_patch(goal: str = '', constraints: dict | None = None) -> dict:
    c = constraints or {}
    low_goal = str(goal or '').lower()
    patch = {}
    if c.get('memory_first'):
        patch = {'training': {'batch': 8, 'imgsz': 512, 'amp': True}}
    elif c.get('accuracy_first'):
        patch = {'training': {'epochs': 200}, 'distillation': {'w_kd': 0.6, 'w_focal': 0.35}}
    elif c.get('speed_first'):
        patch = {'training': {'imgsz': 512, 'batch': 8}, 'distillation': {'T_max': 5.0}}
    elif any(x in low_goal for x in ('memory', '显存', 'oom')):
        patch = {'training': {'batch': 8, 'imgsz': 512, 'amp': True}}
    elif any(x in low_goal for x in ('speed', 'latency', '时延', '吞吐')):
        patch = {'training': {'imgsz': 512, 'batch': 8}, 'distillation': {'T_max': 5.0}}
    elif any(x in low_goal for x in ('accuracy', 'map', '精度')):
        patch = {'training': {'epochs': 200}, 'distillation': {'w_kd': 0.6, 'w_focal': 0.35}}
    else:
        # 兜底仅触及蒸馏侧 kd 权重，避免无端带上 training 等未讨论字段
        patch = {'distillation': {'w_kd': 0.55}}
    path = get_config_dir() / 'distill_config.yaml'
    base_dbg = _load_yaml_file(path) or {}
    declared_snapshot = copy.deepcopy(patch)
    try:
        merged_dbg = _merge_distill_patch(base_dbg, patch)
        cs_dbg = _leaf_config_diff(base_dbg, merged_dbg)
        if not (cs_dbg.get('paths') or []):
            eps = _single_leaf_epsilon_declared_patch(base_dbg, declared_snapshot)
            if eps:
                patch = eps
                _merge_distill_patch(base_dbg, patch)
    except Exception:
        pass
    return {'goal': goal, 'patch': patch, 'need_approval': True}

def _extract_openai_text(data: Any) -> str:
    if not isinstance(data, dict):
        return str(data or '')
    if isinstance(data.get('output_text'), str):
        return data.get('output_text')
    choices = data.get('choices') or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get('message') or {}
        if isinstance(msg, dict):
            c = msg.get('content')
            if isinstance(c, str):
                return c
    out = data.get('output')
    if isinstance(out, list):
        texts = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get('content')
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get('text'), str):
                        texts.append(part['text'])
        if texts:
            return '\n'.join(texts)
    return json.dumps(data, ensure_ascii=False)

def _extract_openai_reasoning_from_message(data: Any) -> str:
    """非流式响应中 message 里的思考文本（若厂商提供）。"""
    if not isinstance(data, dict):
        return ''
    choices = data.get('choices') or []
    if not choices or not isinstance(choices[0], dict):
        return ''
    msg = choices[0].get('message') or {}
    if not isinstance(msg, dict):
        return ''
    for key in ('reasoning_content', 'reasoning'):
        v = msg.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ''

def _openai_compatible_resolve_url(payload: AgentModelInvokeRequest) -> str:
    base = str(payload.api_url or '').strip().rstrip('/')
    endpoint = str(payload.endpoint or '').strip()
    if endpoint:
        return endpoint if endpoint.startswith('http') else f"{base}/{endpoint.lstrip('/')}"
    # 方舟 OpenAI 兼容基地址通常为 /api/v3，默认补全为 /chat/completions。
    if re.search(r'ark\.[^/]+/api/v\d+$', base):
        return f"{base}/chat/completions"
    if base.endswith('/chat/completions') or base.endswith('/responses'):
        return base
    return f"{base}/v1/chat/completions"

def _openai_compatible_build_body(payload: AgentModelInvokeRequest, stream: bool = False) -> dict:
    body_messages = [{'role': m.role, 'content': m.content} for m in payload.messages]
    if payload.system_prompt:
        body_messages = [{'role': 'system', 'content': payload.system_prompt}] + body_messages
    req_body = {
        'model': payload.model or 'gpt-4o-mini',
        'messages': body_messages,
        'temperature': float(payload.temperature),
    }
    if payload.max_tokens is not None:
        req_body['max_tokens'] = int(payload.max_tokens)
    if stream:
        req_body['stream'] = True
    return req_body

def _delta_text_piece(val: Any) -> str:
    if val is None:
        return ''
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, dict):
                if isinstance(item.get('text'), str):
                    parts.append(item['text'])
                elif isinstance(item.get('content'), str):
                    parts.append(item['content'])
            elif isinstance(item, str):
                parts.append(item)
        return ''.join(parts)
    return str(val)

def _delta_reasoning_from_dict(d: dict) -> str:
    if not isinstance(d, dict):
        return ''
    for key in ('reasoning_content', 'reasoning', 'thinking'):
        v = d.get(key)
        if v is None or v == '':
            continue
        return _delta_text_piece(v)
    return ''

def _parse_openai_sse_chunk_json(data_str: str) -> tuple[str, str]:
    """从一条 SSE data 行解析 (content_delta, reasoning_delta)。"""
    if not data_str or data_str.strip() == '[DONE]':
        return '', ''
    try:
        obj = json.loads(data_str)
    except json.JSONDecodeError:
        return '', ''
    choices = obj.get('choices') or []
    if not choices or not isinstance(choices[0], dict):
        return '', ''
    ch0 = choices[0]
    delta = ch0.get('delta') if isinstance(ch0.get('delta'), dict) else {}
    content = _delta_text_piece(delta.get('content')) if delta else ''
    reasoning = _delta_reasoning_from_dict(delta) if delta else ''
    if not content and not reasoning:
        msg = ch0.get('message')
        if isinstance(msg, dict):
            content = _delta_text_piece(msg.get('content'))
            reasoning = _delta_reasoning_from_dict(msg)
    return content, reasoning

def _http_stream_post_openai(url: str, payload: dict, headers: dict | None, timeout: float):
    req_headers = {'Content-Type': 'application/json'}
    if isinstance(headers, dict):
        req_headers.update({str(k): str(v) for k, v in headers.items() if k})
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url=url, data=data, headers=req_headers, method='POST')
    return urllib.request.urlopen(req, timeout=timeout)

def _generate_openai_compatible_sse(payload: AgentModelInvokeRequest):
    url = _openai_compatible_resolve_url(payload)
    req_body = _openai_compatible_build_body(payload, stream=True)
    headers = _set_auth_headers(payload.extra_headers, payload.api_key, prefer_x_api_key=True)
    full_reply: list[str] = []
    full_reasoning: list[str] = []
    try:
        resp = _http_stream_post_openai(url, req_body, headers, float(payload.timeout_sec))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode('utf-8', errors='replace')
        except Exception:
            err_body = str(e)
        yield f"data: {json.dumps({'t': 'error', 'message': f'HTTP {e.code}: {err_body[:2000]}'}, ensure_ascii=False)}\n\n"
        return
    except Exception as e:
        yield f"data: {json.dumps({'t': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        return

    line_buf = b''
    try:
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            line_buf += chunk
            while b'\n' in line_buf:
                raw_line, line_buf = line_buf.split(b'\n', 1)
                line = raw_line.decode('utf-8', errors='replace').rstrip('\r')
                if not line.strip():
                    continue
                if line.startswith(':'):
                    continue
                if not line.startswith('data:'):
                    continue
                data_part = line[5:].lstrip()
                if data_part.strip() == '[DONE]':
                    continue
                c_delta, r_delta = _parse_openai_sse_chunk_json(data_part)
                if c_delta:
                    full_reply.append(c_delta)
                    yield f"data: {json.dumps({'t': 'content', 'd': c_delta}, ensure_ascii=False)}\n\n"
                if r_delta:
                    full_reasoning.append(r_delta)
                    yield f"data: {json.dumps({'t': 'reasoning', 'd': r_delta}, ensure_ascii=False)}\n\n"
        if line_buf.strip():
            line = line_buf.decode('utf-8', errors='replace').rstrip('\r\n')
            if line.startswith('data:'):
                data_part = line[5:].lstrip()
                if data_part.strip() and data_part.strip() != '[DONE]':
                    c_delta, r_delta = _parse_openai_sse_chunk_json(data_part)
                    if c_delta:
                        full_reply.append(c_delta)
                        yield f"data: {json.dumps({'t': 'content', 'd': c_delta}, ensure_ascii=False)}\n\n"
                    if r_delta:
                        full_reasoning.append(r_delta)
                        yield f"data: {json.dumps({'t': 'reasoning', 'd': r_delta}, ensure_ascii=False)}\n\n"
    finally:
        try:
            resp.close()
        except Exception:
            pass

    reply = ''.join(full_reply)
    reasoning = ''.join(full_reasoning)
    yield f"data: {json.dumps({'t': 'done', 'reply': reply, 'reasoning': reasoning}, ensure_ascii=False)}\n\n"

def _invoke_model_openai_compatible(payload: AgentModelInvokeRequest) -> dict:
    url = _openai_compatible_resolve_url(payload)
    req_body = _openai_compatible_build_body(payload, stream=False)
    headers = _set_auth_headers(payload.extra_headers, payload.api_key, prefer_x_api_key=True)
    raw = _http_json_request('POST', url=url, payload=req_body, headers=headers, timeout=float(payload.timeout_sec))
    out: dict = {'reply': _extract_openai_text(raw), 'raw': raw, 'provider': 'openai_compatible'}
    rsn = _extract_openai_reasoning_from_message(raw)
    if rsn:
        out['reasoning'] = rsn
    return out

def _invoke_model_anthropic(payload: AgentModelInvokeRequest) -> dict:
    base = str(payload.api_url or '').strip().rstrip('/')
    endpoint = str(payload.endpoint or '').strip()
    url = endpoint if endpoint else (f"{base}/v1/messages" if not base.endswith('/messages') else base)
    req_body = {
        'model': payload.model or 'claude-3-5-sonnet-latest',
        'messages': [{'role': m.role, 'content': m.content} for m in payload.messages],
        'max_tokens': int(payload.max_tokens or 1024),
        'temperature': float(payload.temperature),
    }
    if payload.system_prompt:
        req_body['system'] = payload.system_prompt
    headers = {'anthropic-version': '2023-06-01'}
    headers.update(_set_auth_headers(payload.extra_headers, payload.api_key, prefer_x_api_key=True))
    raw = _http_json_request('POST', url=url, payload=req_body, headers=headers, timeout=float(payload.timeout_sec))
    text = ''
    content = raw.get('content') if isinstance(raw, dict) else None
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get('text'), str):
                text += part.get('text', '')
    return {'reply': text or json.dumps(raw, ensure_ascii=False), 'raw': raw, 'provider': 'anthropic_style'}

def _invoke_model_custom(payload: AgentModelInvokeRequest) -> dict:
    headers = _set_auth_headers(payload.extra_headers, payload.api_key, prefer_x_api_key=True)
    req_body = {
        'model': payload.model,
        'messages': [{'role': m.role, 'content': m.content} for m in payload.messages],
        'system_prompt': payload.system_prompt,
        'temperature': float(payload.temperature),
        'max_tokens': payload.max_tokens,
    }
    raw = _http_json_request('POST', url=payload.api_url, payload=req_body, headers=headers, timeout=float(payload.timeout_sec))
    if isinstance(raw, dict):
        reply = raw.get('reply') or raw.get('message') or raw.get('output') or ''
        return {'reply': str(reply), 'raw': raw, 'provider': 'custom'}
    return {'reply': str(raw), 'raw': raw, 'provider': 'custom'}

def _agent_invoke_model(payload: AgentModelInvokeRequest) -> dict:
    provider = str(payload.provider or '').strip().lower()
    if provider in ('openai_compatible', 'openai', 'openai-compatible'):
        return _invoke_model_openai_compatible(payload)
    if provider in ('anthropic', 'anthropic_style', 'claude'):
        return _invoke_model_anthropic(payload)
    if provider in ('custom', 'custom_adapter'):
        return _invoke_model_custom(payload)
    raise ValueError(f'不支持的 provider: {provider}')

def agent_patch_validate(payload: AgentPatchValidateRequest):
    result = _agent_validate_patch(payload.patch, strict=bool(payload.strict))
    return {'status': 'ok', **result}

def agent_patch_preview(payload: AgentPatchPreviewRequest):
    """合并 patch 到 distill_config.yaml 的内存预览，并签发 run 维度审批令牌。"""
    _prune_agent_patch_store()
    patch = payload.patch
    if not isinstance(patch, dict) or not patch:
        return _error('patch 必须为非空对象', 400)
    valid_result = _agent_validate_patch(patch, strict=True)
    if not valid_result.get('valid'):
        return _error('; '.join(valid_result.get('errors') or ['patch 不合法']), 400)
    path = get_config_dir() / 'distill_config.yaml'
    base = _load_yaml_file(path) or {}
    try:
        merged = _merge_distill_patch(base, patch)
    except ValueError as e:
        return _error(str(e), 400)
    except Exception as e:
        return _error(str(e), 500)
    tok = str(uuid.uuid4())
    run_id = _safe_run_id(payload.run_id)
    req_hash = _agent_request_hash(run_id, patch)
    _agent_patch_store[tok] = {
        'patch': copy.deepcopy(patch),
        'base': copy.deepcopy(base),
        'merged': merged,
        'run_id': run_id,
        'operator': payload.operator or 'user',
        'reason': payload.reason or '',
        'request_hash': req_hash,
        'used': False,
        'expires': time.time() + _AGENT_PATCH_TTL,
    }
    patch_yaml = yaml.dump(patch, allow_unicode=True, default_flow_style=False, sort_keys=False)
    change_summary_raw = _leaf_config_diff(base, merged)
    change_summary = _filter_change_summary_to_patch_declared(change_summary_raw, patch)
    return {
        'status': 'ok',
        'approval_token': tok,
        'approval_ticket_id': tok,
        'expires_in_sec': int(_AGENT_PATCH_TTL),
        'run_id': run_id,
        'request_hash': req_hash,
        'patch_yaml': patch_yaml,
        'merged_preview': merged,
        'validation': valid_result,
        'change_summary': change_summary,
    }

def agent_patch_apply(payload: AgentPatchApplyRequest):
    """凭审批令牌将预览中的合并结果写入 configs/distill_config.yaml，并记录 run 历史。"""
    
    _prune_agent_patch_store()
    tok = payload.approval_token or payload.token
    if not isinstance(tok, str) or not tok:
        return _error('缺少 approval_token', 400)
    rec = _agent_patch_store.get(tok)
    if not rec or rec.get('expires', 0) < time.time():
        return _error('审批令牌无效或已过期，请重新预览', 400)
    if rec.get('used'):
        return _error('审批令牌已使用，禁止重放', 400)
    expect_run_id = _safe_run_id(payload.run_id)
    if expect_run_id != rec.get('run_id'):
        return _error('run_id 与审批票据不匹配', 400)
    if payload.request_hash and payload.request_hash != rec.get('request_hash'):
        return _error('request_hash 校验失败，拒绝写入', 400)
    merged = rec.get('merged')
    if not isinstance(merged, dict):
        return _error('内部数据损坏', 500)
    base_cfg = rec.get('base') if isinstance(rec.get('base'), dict) else {}
    out_path = get_config_dir() / 'distill_config.yaml'
    try:
        _save_yaml_file(out_path, merged)
        backend_state.last_saved_config = {'name': 'distill_config.yaml', 'config': merged}
        rec['used'] = True
        _agent_patch_store.pop(tok, None)
        hist_rec = _agent_record_history(
            run_id=expect_run_id,
            before_cfg=base_cfg,
            after_cfg=merged,
            operator=payload.operator or rec.get('operator') or 'user',
            reason=payload.reason or rec.get('reason') or '',
            action='apply_patch',
        )
        try:
            file_mtime_ns = int(out_path.stat().st_mtime_ns)
        except OSError:
            file_mtime_ns = 0
        return {
            'status': 'ok',
            'message': '已写入 configs/distill_config.yaml',
            'config': merged,
            'run_id': expect_run_id,
            'history_version': hist_rec.get('version'),
            'file_mtime_ns': file_mtime_ns,
        }
    except Exception as e:
        return _error(str(e), 500)

def agent_run_history(run_id: str):
    rid = _safe_run_id(run_id)
    hist = _agent_load_history(rid)
    compact = []
    for rec in reversed(hist):
        compact.append({
            'version': rec.get('version'),
            'timestamp': rec.get('timestamp'),
            'operator': rec.get('operator'),
            'reason': rec.get('reason'),
            'action': rec.get('action'),
        })
    return {'status': 'ok', 'run_id': rid, 'window_size': _AGENT_MAX_ROLLBACKS_PER_RUN, 'history': compact}

def agent_run_rollback(run_id: str, payload: AgentRunHistoryRollbackRequest):
    
    rid = _safe_run_id(run_id or payload.run_id)
    hist = _agent_load_history(rid)
    if not hist:
        return _error('该 run 没有可回退历史', 404)
    target = None
    if payload.target_version is not None:
        for item in hist:
            if int(item.get('version', -1)) == int(payload.target_version):
                target = item
                break
    else:
        steps = int(payload.steps or 1)
        if steps <= 0:
            return _error('steps 必须 >= 1', 400)
        if steps > len(hist):
            return _error('可回退步数超过历史窗口', 400)
        target = hist[-steps]
    if not isinstance(target, dict):
        return _error('未找到目标版本', 404)
    restore_cfg = target.get('after_config')
    if not isinstance(restore_cfg, dict):
        return _error('目标历史数据损坏', 500)
    out_path = get_config_dir() / 'distill_config.yaml'
    current_cfg = _load_yaml_file(out_path) or {}
    try:
        _save_yaml_file(out_path, restore_cfg)
        backend_state.last_saved_config = {'name': 'distill_config.yaml', 'config': restore_cfg}
        new_rec = _agent_record_history(
            run_id=rid,
            before_cfg=current_cfg,
            after_cfg=restore_cfg,
            operator=payload.operator,
            reason=payload.reason,
            action=f'rollback_to_v{target.get("version")}',
        )
        try:
            file_mtime_ns = int(out_path.stat().st_mtime_ns)
        except OSError:
            file_mtime_ns = 0
        return {
            'status': 'ok',
            'run_id': rid,
            'rolled_back_to_version': target.get('version'),
            'history_version': new_rec.get('version'),
            'config': restore_cfg,
            'file_mtime_ns': file_mtime_ns,
        }
    except Exception as e:
        return _error(str(e), 500)

def agent_tools_contract():
    return {
        'status': 'ok',
        'version': AGENT_TOOL_CONTRACT_VERSION,
        'philosophy': 'single-purpose tools, JSON in/out, composable pipeline, explicit approval gate',
        'tools': [
            {'name': 'agent.get_context', 'input': {'run_id': 'str', 'config_name': 'str?'}, 'output': 'context'},
            {'name': 'agent.analyze_params', 'input': {'run_id': 'str', 'objective': 'str', 'config_name': 'str?'}, 'output': 'analysis'},
            {'name': 'agent.propose_patch', 'input': {'goal': 'str', 'constraints': 'dict?'}, 'output': 'patch'},
            {'name': 'agent.validate_patch', 'input': {'patch': 'dict', 'strict': 'bool?'}, 'output': 'validation'},
            {'name': 'agent.preview_patch', 'input': {'run_id': 'str', 'patch': 'dict', 'operator': 'str?', 'reason': 'str?'}, 'output': 'approval_ticket'},
            {'name': 'agent.apply_patch_with_approval', 'input': {'run_id': 'str', 'approval_token': 'str', 'request_hash': 'str?'}, 'output': 'applied_config'},
            {'name': 'agent.list_run_history', 'input': {'run_id': 'str'}, 'output': 'history'},
            {'name': 'agent.rollback_run_config', 'input': {'run_id': 'str', 'target_version': 'int?'}, 'output': 'rolled_back_config'},
        ],
        'error_codes': {
            'E_PATCH_INVALID': 400,
            'E_APPROVAL_EXPIRED': 400,
            'E_APPROVAL_REPLAY': 400,
            'E_RUN_HISTORY_EMPTY': 404,
            'E_INTERNAL': 500,
        },
    }

def _normalize_agent_tool_name(name: Any) -> str:
    raw = str(name or '').strip()
    if not raw:
        return ''
    lowered = raw.lower().replace('-', '_').replace(' ', '_')
    compact = lowered.replace('.', '').replace('_', '')
    alias = {
        'agent.get_context': 'agent.get_context',
        'agent.analyze_params': 'agent.analyze_params',
        'agent.propose_patch': 'agent.propose_patch',
        'agent.validate_patch': 'agent.validate_patch',
        'agent.preview_patch': 'agent.preview_patch',
        'agent.apply_patch_with_approval': 'agent.apply_patch_with_approval',
        'agent.list_run_history': 'agent.list_run_history',
        'agent.rollback_run_config': 'agent.rollback_run_config',
        'get_context': 'agent.get_context',
        'analyze_params': 'agent.analyze_params',
        'propose_patch': 'agent.propose_patch',
        'validate_patch': 'agent.validate_patch',
        'preview_patch': 'agent.preview_patch',
        'previewpatch': 'agent.preview_patch',
        'apply_patch_with_approval': 'agent.apply_patch_with_approval',
        'list_run_history': 'agent.list_run_history',
        'rollback_run_config': 'agent.rollback_run_config',
        'agentpreviewpatch': 'agent.preview_patch',
    }
    if lowered in alias:
        return alias[lowered]
    if compact in alias:
        return alias[compact]
    return raw

def agent_tools_execute(payload: AgentToolExecuteRequest):
    tool = _normalize_agent_tool_name(payload.tool)
    args = payload.args or {}
    try:
        if tool == 'agent.get_context':
            return {'status': 'ok', 'tool': tool, 'result': _agent_get_context(args.get('run_id', 'default'), args.get('config_name', 'distill_config.yaml'))}
        if tool == 'agent.analyze_params':
            return {'status': 'ok', 'tool': tool, 'result': _agent_analyze_params(args.get('run_id', 'default'), args.get('objective', ''), args.get('config_name', 'distill_config.yaml'))}
        if tool == 'agent.propose_patch':
            return {'status': 'ok', 'tool': tool, 'result': _agent_propose_patch(args.get('goal', ''), args.get('constraints') or {})}
        if tool == 'agent.validate_patch':
            return {'status': 'ok', 'tool': tool, 'result': _agent_validate_patch(args.get('patch') or {}, bool(args.get('strict', True)))}
        if tool == 'agent.preview_patch':
            req = AgentPatchPreviewRequest(
                patch=args.get('patch') or {},
                run_id=args.get('run_id', 'default'),
                operator=args.get('operator', 'agent'),
                reason=args.get('reason', ''),
            )
            return agent_patch_preview(req)
        if tool == 'agent.apply_patch_with_approval':
            req = AgentPatchApplyRequest(
                approval_token=args.get('approval_token') or args.get('token'),
                run_id=args.get('run_id', 'default'),
                operator=args.get('operator', 'agent'),
                request_hash=args.get('request_hash'),
                reason=args.get('reason', ''),
            )
            return agent_patch_apply(req)
        if tool == 'agent.list_run_history':
            return agent_run_history(args.get('run_id', 'default'))
        if tool == 'agent.rollback_run_config':
            req = AgentRunHistoryRollbackRequest(
                run_id=args.get('run_id', 'default'),
                target_version=args.get('target_version'),
                steps=args.get('steps'),
                operator=args.get('operator', 'agent'),
                reason=args.get('reason', 'manual rollback'),
            )
            return agent_run_rollback(args.get('run_id', 'default'), req)
        return _error(f'未知工具: {tool}', 404)
    except Exception as e:
        return _error(f'{tool} 执行失败: {e}', 500)

def agent_model_invoke(payload: AgentModelInvokeRequest):
    try:
        result = _agent_invoke_model(payload)
        return {'status': 'ok', **result}
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8', errors='replace')
        except Exception:
            body = str(e)
        return _error(f'模型调用失败(HTTP {e.code}): {body}', 502)
    except Exception as e:
        return _error(f'模型调用失败: {e}', 500)

def agent_model_invoke_stream(payload: AgentModelInvokeRequest):
    """浏览器 HTTPS Agent 走本地中继时的 SSE：与前端 readAgentInvokeSseStream 约定一致（见 _generate_openai_compatible_sse）。"""
    provider = str(payload.provider or '').strip().lower()
    if provider not in ('openai_compatible', 'openai', 'openai-compatible'):
        return _error('invoke-stream 当前仅支持 openai_compatible', 400)
    return StreamingResponse(
        _generate_openai_compatible_sse(payload),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )
