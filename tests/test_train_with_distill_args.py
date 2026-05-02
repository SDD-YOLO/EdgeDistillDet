from pathlib import Path

from scripts.train_with_distill import _build_train_args


def test_build_train_args_clamps_non_positive_max_det_to_default():
    train_cfg = {
        "data_yaml": "configs/dataset_coco128.yaml",
        "epochs": 1,
        "imgsz": 320,
        "batch": 4,
        "max_det": 0,
    }
    output_cfg = {"project": "runs", "name": "exp_test"}

    args = _build_train_args(
        train_cfg=train_cfg,
        output_cfg=output_cfg,
        resume_path=None,
        root=Path("D:/Personal_Files/Projects/Github/EdgeDistillDet"),
        allow_overwrite=True,
    )

    assert args["max_det"] == 300
