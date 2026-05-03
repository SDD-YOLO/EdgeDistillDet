from web.services import backend_common, backend_train


def test_output_check_next_exp_after_exp10(tmp_path, monkeypatch):
    base_dir = tmp_path / "repo"
    runs_dir = base_dir / "runs"
    runs_dir.mkdir(parents=True)

    for run_name in ["exp1", "exp2", "exp10"]:
        (runs_dir / run_name).mkdir()

    monkeypatch.setattr(backend_common, "BASE_DIR", base_dir)
    monkeypatch.setattr(backend_train, "BASE_DIR", base_dir)

    result = backend_train.output_check("runs")

    assert result["status"] == "ok"
    assert result["next_exp_name"] == "exp11"
