from __future__ import annotations

from types import SimpleNamespace

from core.distillation.adaptive_kd_trainer import AdaptiveKDTrainer


def test_fit_epoch_end_injects_distill_metrics_and_updates_csv(tmp_path):
    trainer = object.__new__(AdaptiveKDTrainer)
    trainer.save_dir = tmp_path
    trainer._distill_log = [
        {
            "epoch": 1,
            "phase": "warm",
            "task_loss": 1.2,
            "kd_loss": 0.4,
            "alpha": 0.55,
            "temperature": 3.5,
        }
    ]

    results_csv = tmp_path / "results.csv"
    results_csv.write_text("epoch,time,metrics/mAP50\n1,0.1,0.42\n", encoding="utf-8")

    fit_trainer = SimpleNamespace(
        epoch=0,
        metrics={"metrics/mAP50": 0.42},
        results_dict={"metrics/mAP50": 0.42},
    )

    trainer._on_fit_epoch_end(fit_trainer)

    assert fit_trainer.metrics["distill/alpha"] == 0.55
    assert fit_trainer.metrics["distill/temperature"] == 3.5
    assert fit_trainer.metrics["distill/kd_loss"] == 0.4
    assert fit_trainer.results_dict["distill/alpha"] == 0.55

    content = results_csv.read_text(encoding="utf-8")
    assert "distill/alpha" not in content
    assert "distill/temperature" not in content
    assert "distill/kd_loss" not in content
    assert "0.42" in content


def test_train_end_rewrites_results_csv_with_distill_columns(tmp_path):
    trainer = object.__new__(AdaptiveKDTrainer)
    trainer.save_dir = tmp_path
    trainer._distill_log = [
        {
            "epoch": 1,
            "phase": "warm",
            "task_loss": 1.2,
            "kd_loss": 0.4,
            "alpha": 0.55,
            "temperature": 3.5,
        }
    ]

    results_csv = tmp_path / "results.csv"
    results_csv.write_text("epoch,time,metrics/mAP50\n1,0.1,0.42\n", encoding="utf-8")

    trainer._on_train_end(SimpleNamespace(save_dir=tmp_path))

    content = results_csv.read_text(encoding="utf-8")
    assert "distill/alpha" in content
    assert "distill/temperature" in content
    assert "distill/kd_loss" in content
    assert "0.55" in content
