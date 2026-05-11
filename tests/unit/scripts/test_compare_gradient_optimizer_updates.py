import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "compare_gradient_optimizer_updates.py"
SPEC = importlib.util.spec_from_file_location("compare_gradient_optimizer_updates", SCRIPT_PATH)
compare_gradient_optimizer_updates = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = compare_gradient_optimizer_updates
SPEC.loader.exec_module(compare_gradient_optimizer_updates)


def _metadata(step: int = 0) -> dict[str, object]:
    return {
        "step": step,
        "batch_hash": "batch",
        "pre_step_state_hash": "state",
    }


def _write_step(
    root: Path,
    step: int,
    gradients: dict[str, torch.Tensor],
    trainable_state: dict[str, torch.Tensor] | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    step_dir = root / f"step_{step:06d}"
    step_dir.mkdir(parents=True)
    torch.save(gradients, step_dir / "gradients.pt")
    if trainable_state is not None:
        torch.save(trainable_state, step_dir / "trainable_state.pt")
    with open(step_dir / "metadata.json", "w") as f:
        json.dump(metadata or _metadata(step), f)


def test_adamw_delta_matches_torch_optimizer_single_step() -> None:
    name = "model.layers.0.self_attn.q_proj.lora_A.0"
    param = torch.tensor([1.0, -2.0], dtype=torch.float64)
    grad = torch.tensor([0.2, -0.4], dtype=torch.float64)
    config = compare_gradient_optimizer_updates.AdamWConfig(lr=0.1, weight_decay=0.01, max_norm=None)
    moments = compare_gradient_optimizer_updates._init_moments([name], {name: grad})

    deltas, _, _ = compare_gradient_optimizer_updates._adamw_update_delta(
        {name: param},
        {name: grad},
        moments,
        [name],
        optimizer_step=1,
        config=config,
    )

    torch_param = torch.nn.Parameter(param.clone())
    optimizer = torch.optim.AdamW(
        [torch_param],
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    torch_param.grad = grad.clone()
    before = torch_param.detach().clone()
    optimizer.step()

    torch.testing.assert_close(deltas[name], torch_param.detach() - before)


def test_gradient_clipping_scales_adamw_gradient_term() -> None:
    name = "model.layers.0.self_attn.q_proj.lora_A.0"
    gradients = {name: torch.tensor([3.0, 4.0], dtype=torch.float64)}
    config = compare_gradient_optimizer_updates.AdamWConfig(lr=0.1, weight_decay=0.0, max_norm=1.0)
    moments = compare_gradient_optimizer_updates._init_moments([name], gradients)

    _, grad_norm, clip_coef = compare_gradient_optimizer_updates._adamw_update_delta(
        {name: torch.zeros(2, dtype=torch.float64)},
        gradients,
        moments,
        [name],
        optimizer_step=1,
        config=config,
    )

    assert grad_norm == pytest.approx(5.0)
    assert clip_coef == pytest.approx(0.2)


def test_method_specific_state_uses_each_method_history() -> None:
    name = "model.layers.0.self_attn.q_proj.lora_A.0"
    config = compare_gradient_optimizer_updates.AdamWConfig(lr=0.1, weight_decay=0.0, max_norm=None)
    state = {name: torch.tensor([1.0], dtype=torch.float64)}
    ref = {
        0: {name: torch.tensor([1.0], dtype=torch.float64)},
        1: {name: torch.tensor([1.0], dtype=torch.float64)},
    }
    candidate = {
        0: {name: torch.tensor([10.0], dtype=torch.float64)},
        1: {name: torch.tensor([1.0], dtype=torch.float64)},
    }

    ref_moments = compare_gradient_optimizer_updates._init_moments([name], ref[0])
    candidate_moments = compare_gradient_optimizer_updates._init_moments([name], ref[0])
    compare_gradient_optimizer_updates._adamw_update_delta(state, ref[0], ref_moments, [name], 1, config)
    compare_gradient_optimizer_updates._adamw_update_delta(state, candidate[0], candidate_moments, [name], 1, config)
    shared_delta, _, _ = compare_gradient_optimizer_updates._adamw_update_delta(
        state,
        candidate[1],
        compare_gradient_optimizer_updates._copy_moments(ref_moments),
        [name],
        2,
        config,
    )
    method_delta, _, _ = compare_gradient_optimizer_updates._adamw_update_delta(
        state, candidate[1], candidate_moments, [name], 2, config
    )

    assert not torch.allclose(shared_delta[name], method_delta[name])


def test_rejects_non_lora_params_by_default() -> None:
    with pytest.raises(ValueError, match="non-LoRA"):
        compare_gradient_optimizer_updates._validate_lora_names(["model.embed_tokens.weight"])


def test_cli_writes_incremental_optimizer_update_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    name = "model.layers.0.self_attn.q_proj.lora_A.0"
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    output = tmp_path / "updates.json"
    trainable_state = {name: torch.tensor([1.0, 2.0], dtype=torch.float64)}
    _write_step(reference, 0, {name: torch.tensor([1.0, 2.0])}, trainable_state=trainable_state)
    _write_step(candidate, 0, {name: torch.tensor([1.0, 1.5])})
    _write_step(baseline, 0, {name: torch.tensor([0.0, 2.25])})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_gradient_optimizer_updates.py",
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

    compare_gradient_optimizer_updates.main()

    stdout = capsys.readouterr().out
    results = json.loads(output.read_text())
    assert "shared_state_relative_l2_existing shared_state_relative_l2_fix" in stdout
    assert len(results) == 1
    assert results[0]["compared"] is True
    assert results[0]["parameter_count"] == 1
    assert results[0]["optimizer"]["weight_decay"] == 0.01
    assert "method_state_relative_l2_fix" in results[0]
