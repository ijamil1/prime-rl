import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "compare_gradient_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("compare_gradient_diagnostics", SCRIPT_PATH)
compare_gradient_diagnostics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare_gradient_diagnostics)


def _metadata(step: int = 0) -> dict[str, object]:
    return {
        "step": step,
        "batch_hash": "batch",
        "pre_step_state_hash": "state",
        "dp_loss_scale_std": 1.0,
        "dp_loss_scale_min": 2.0,
        "dp_loss_scale_max": 3.0,
    }


def _write_step(root: Path, gradients: dict[str, torch.Tensor], metadata: dict[str, object] | None = None) -> None:
    step_dir = root / "step_000000"
    step_dir.mkdir(parents=True)
    torch.save(gradients, step_dir / "gradients.pt")
    with open(step_dir / "metadata.json", "w") as f:
        json.dump(metadata or _metadata(), f)


def test_element_and_parameter_win_rates() -> None:
    reference = {
        "p1": torch.tensor([0.0, 10.0]),
        "p2": torch.tensor([1.0]),
    }
    baseline = {
        "p1": torch.tensor([1.0, 8.0]),
        "p2": torch.tensor([2.0]),
    }
    candidate = {
        "p1": torch.tensor([0.5, 7.0]),
        "p2": torch.tensor([1.1]),
    }
    names = ["p1", "p2"]
    reference_flat = compare_gradient_diagnostics._flatten(reference, names)
    baseline_flat = compare_gradient_diagnostics._flatten(baseline, names)
    candidate_flat = compare_gradient_diagnostics._flatten(candidate, names)

    element_metrics = compare_gradient_diagnostics._element_win_metrics(
        reference_flat, candidate_flat, baseline_flat, atol=0.0
    )
    parameter_metrics = compare_gradient_diagnostics._parameter_win_metrics(
        reference, candidate, baseline, names, atol=0.0
    )

    assert element_metrics["candidate_element_win_count"] == 2
    assert element_metrics["baseline_element_win_count"] == 1
    assert element_metrics["element_tie_count"] == 0
    assert element_metrics["candidate_element_win_rate"] == pytest.approx(2 / 3)
    assert parameter_metrics["candidate_param_win_count"] == 1
    assert parameter_metrics["baseline_param_win_count"] == 1
    assert parameter_metrics["param_tie_count"] == 0
    assert parameter_metrics["candidate_param_win_rate"] == pytest.approx(0.5)


def test_win_rate_tie_atol() -> None:
    counts = compare_gradient_diagnostics._win_counts(
        candidate_error=torch.tensor([1.0, 2.0, 3.0]),
        baseline_error=torch.tensor([1.1, 1.0, 3.0]),
        atol=0.2,
    )

    assert counts["candidate_wins"] == 0
    assert counts["baseline_wins"] == 1
    assert counts["ties"] == 2


def test_validate_gradients_rejects_mismatched_names_and_shapes() -> None:
    reference = {"p1": torch.zeros(2)}

    with pytest.raises(ValueError, match="names differ"):
        compare_gradient_diagnostics._validate_gradients(reference, {"p2": torch.zeros(2)}, "candidate")

    with pytest.raises(ValueError, match="shape differs"):
        compare_gradient_diagnostics._validate_gradients(reference, {"p1": torch.zeros(3)}, "candidate")


def test_three_way_cli_writes_win_rate_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    output = tmp_path / "comparison.json"
    _write_step(reference, {"p": torch.tensor([1.0, 2.0])})
    _write_step(candidate, {"p": torch.tensor([1.0, 1.5])})
    _write_step(baseline, {"p": torch.tensor([0.0, 2.25])})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_gradient_diagnostics.py",
            str(reference),
            str(candidate),
            "--baseline-candidate-dir",
            str(baseline),
            "--candidate-label",
            "fix",
            "--baseline-label",
            "existing",
            "--json-output",
            str(output),
        ],
    )

    compare_gradient_diagnostics.main()

    stdout = capsys.readouterr().out
    results = json.loads(output.read_text())
    assert "cosine_existing cosine_fix" in stdout
    assert len(results) == 1
    assert results[0]["compared"] is True
    assert results[0]["candidate_element_win_count"] == 1
    assert results[0]["baseline_element_win_count"] == 1
    assert results[0]["candidate_param_win_rate"] == 1.0
    assert "relative_l2_fix" in results[0]
    assert "relative_l2_existing" in results[0]
