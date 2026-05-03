from __future__ import annotations

import copy

import torch

from core.distillation.adaptive_kd_trainer import (
    AdaptiveKDTrainer,
    _DistillCriterionProxy,
    _move_detection_loss_to_device,
)


class _DummyCriterion:
    def __call__(self, preds, batch):
        return torch.tensor(2.0), torch.tensor([1.0, 2.0])

    def update(self, *args, **kwargs):
        return "updated"


class _DummyModel:
    def __init__(self):
        self.criterion = None
        self.nc = 3

    def init_criterion(self):
        return _DummyCriterion()


def test_distill_proxy_installs_on_model_and_counts_batches():
    trainer = object.__new__(AdaptiveKDTrainer)
    trainer.model = _DummyModel()
    trainer.teacher_model = object()
    trainer._warm_epochs = 5
    trainer._distill_entered = False
    trainer._epoch_task_loss = 0.0
    trainer._epoch_kd_loss = 0.0
    trainer._batch_count = 0
    trainer._alpha_scheduler = None
    trainer._temp_scheduler = None
    trainer._distill_loss = None
    trainer._base_criterion = None
    trainer._criterion_proxy = None
    trainer.epoch = 0

    trainer._install_distill_criterion()

    assert isinstance(trainer.model.criterion, _DistillCriterionProxy)

    batch = {"img": torch.zeros(1, 3, 8, 8)}
    loss, loss_items = trainer.model.criterion(torch.zeros(1, 3), batch)

    assert torch.is_tensor(loss)
    assert torch.equal(loss, torch.tensor(2.0))
    assert torch.equal(loss_items, torch.tensor([1.0, 2.0]))
    assert trainer._batch_count == 1
    assert trainer._epoch_task_loss == 2.0


def test_distill_proxy_populates_missing_model_args():
    trainer = object.__new__(AdaptiveKDTrainer)
    trainer.model = _DummyModel()
    trainer.teacher_model = object()
    trainer._warm_epochs = 5
    trainer._distill_entered = False
    trainer._epoch_task_loss = 0.0
    trainer._epoch_kd_loss = 0.0
    trainer._batch_count = 0
    trainer._alpha_scheduler = None
    trainer._temp_scheduler = None
    trainer._distill_loss = None
    trainer._base_criterion = None
    trainer._criterion_proxy = None
    trainer.epoch = 0
    trainer.args = object()

    trainer._install_distill_criterion()

    assert getattr(trainer.model, "args", None) is trainer.args
    assert isinstance(trainer.model.criterion, _DistillCriterionProxy)


def test_distill_proxy_supports_deepcopy():
    trainer = object.__new__(AdaptiveKDTrainer)
    trainer.model = _DummyModel()
    trainer.teacher_model = object()
    trainer._warm_epochs = 5
    trainer._distill_entered = False
    trainer._epoch_task_loss = 0.0
    trainer._epoch_kd_loss = 0.0
    trainer._batch_count = 0
    trainer._alpha_scheduler = None
    trainer._temp_scheduler = None
    trainer._distill_loss = None
    trainer._base_criterion = None
    trainer._criterion_proxy = None
    trainer.epoch = 0
    trainer.args = object()

    trainer._install_distill_criterion()

    cloned = copy.deepcopy(trainer.model.criterion)

    assert isinstance(cloned, _DistillCriterionProxy)


def test_move_detection_loss_to_device_moves_proj_when_available():
    loss_obj = type("LossObj", (), {})()
    loss_obj.device = torch.device("cpu")
    loss_obj.proj = torch.arange(4)
    loss_obj.bbox_loss = torch.nn.Identity()

    moved = _move_detection_loss_to_device(loss_obj, torch.device("cpu"))

    assert moved.device == torch.device("cpu")
    assert moved.proj.device.type == "cpu"
