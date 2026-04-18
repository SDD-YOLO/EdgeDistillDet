"""agent 配置预览的 change_summary（叶子 diff）逻辑。"""

from __future__ import annotations

from web.services.backend_agent import (
    _filter_change_summary_to_patch_declared,
    _leaf_config_diff,
    _merge_distill_patch,
    _single_leaf_epsilon_declared_patch,
)


def test_leaf_diff_changed_scalar():
    base = {'training': {'lr0': 0.001, 'epochs': 10}}
    merged = {'training': {'lr0': 0.002, 'epochs': 10}}
    out = _leaf_config_diff(base, merged)
    paths = out['paths']
    assert len(paths) == 1
    assert paths[0]['path'] == 'training.lr0'
    assert paths[0]['kind'] == 'changed'
    assert paths[0]['before'] == 0.001
    assert paths[0]['after'] == 0.002
    assert out['stats']['changed'] == 1


def test_leaf_diff_added_nested():
    base = {'training': {'epochs': 1}}
    merged = {'training': {'epochs': 1, 'extra': {'x': 1}}}
    out = _leaf_config_diff(base, merged)
    paths = {p['path']: p for p in out['paths']}
    assert paths['training.extra.x']['kind'] == 'added'
    assert paths['training.extra.x']['after'] == 1


def test_leaf_diff_ignores_other_top_level():
    base = {'training': {'epochs': 1}, 'other': 99}
    merged = {'training': {'epochs': 2}, 'other': 99}
    out = _leaf_config_diff(base, merged)
    assert all(not p['path'].startswith('other') for p in out['paths'])


def test_single_leaf_epsilon_only_declared_keys():
    """模板与磁盘恒等时，仅在声明过的路径上 bump 一个叶子，不引入未声明字段。"""
    base = {
        'distillation': {
            'temperature': 1.1,
            'w_kd': 0.55,
            'T_max': 3,
        },
        'training': {'epochs': 120},
    }
    declared = {'training': {'epochs': 120}, 'distillation': {'w_kd': 0.55}}
    eps = _single_leaf_epsilon_declared_patch(base, declared)
    merged = _merge_distill_patch(base, eps)
    out = _leaf_config_diff(base, merged)
    assert out['stats']['changed'] == 1
    paths = {p['path'] for p in out['paths']}
    assert paths == {'distillation.w_kd'}
    assert 'distillation.temperature' not in paths


def test_single_leaf_epsilon_nested_declared():
    base = {'training': {'cloud_api': {'poll_interval_sec': 3}}}
    declared = {'training': {'cloud_api': {'poll_interval_sec': 3}}}
    eps = _single_leaf_epsilon_declared_patch(base, declared)
    merged = _merge_distill_patch(base, eps)
    out = _leaf_config_diff(base, merged)
    assert len(out['paths']) == 1
    assert out['paths'][0]['path'] == 'training.cloud_api.poll_interval_sec'


def test_filter_change_summary_only_declared_patch_leaves():
    """全集 diff 若含未在 patch 中声明的路径（如伪差异），审批摘要应只保留 patch 叶子。"""
    raw = {
        'paths': [
            {'path': 'distillation.w_kd', 'kind': 'changed', 'before': 0.5, 'after': 0.6},
            {'path': 'distillation.temperature', 'kind': 'changed', 'before': 4, 'after': 4},
        ],
        'stats': {'changed': 2},
    }
    patch = {'distillation': {'w_kd': 0.6}}
    out = _filter_change_summary_to_patch_declared(raw, patch)
    assert len(out['paths']) == 1
    assert out['paths'][0]['path'] == 'distillation.w_kd'
    assert out['stats']['changed'] == 1
