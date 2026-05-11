import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class AdamWConfig:
    lr: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_norm: float | None = 1.0


Moments = dict[str, tuple[Tensor, Tensor]]


def _step_dirs(root: Path) -> dict[int, Path]:
    result = {}
    for path in root.glob("step_*"):
        if path.is_dir():
            result[int(path.name.removeprefix("step_"))] = path
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_tensors(path: Path) -> dict[str, Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_step(root: Path, step: int) -> tuple[dict[str, Tensor], dict[str, Any]]:
    step_dir = _step_dirs(root)[step]
    return _load_tensors(step_dir / "gradients.pt"), _load_json(step_dir / "metadata.json")


def _load_reference_trainable_state(root: Path, step: int) -> dict[str, Tensor]:
    step_dir = _step_dirs(root)[step]
    return _load_tensors(step_dir / "trainable_state.pt")


def _common_steps(*roots: Path) -> list[int]:
    step_sets = [set(_step_dirs(root)) for root in roots]
    common_steps = sorted(set.intersection(*step_sets))
    if not common_steps:
        raise ValueError("No common gradient diagnostic steps found")
    return common_steps


def _validate_tensor_dict(reference: dict[str, Tensor], candidate: dict[str, Tensor], label: str) -> list[str]:
    reference_names = list(reference)
    candidate_names = list(candidate)
    if reference_names != candidate_names:
        missing = sorted(set(reference_names) - set(candidate_names))
        extra = sorted(set(candidate_names) - set(reference_names))
        raise ValueError(f"Tensor names differ for {label}: missing={missing}, extra={extra}")
    for name in reference_names:
        if reference[name].shape != candidate[name].shape:
            raise ValueError(
                f"Tensor shape differs for {label} parameter {name}: "
                f"{tuple(reference[name].shape)} != {tuple(candidate[name].shape)}"
            )
    return reference_names


def _validate_lora_names(names: list[str]) -> None:
    non_lora = [name for name in names if "lora_A" not in name and "lora_B" not in name]
    if non_lora:
        sample = non_lora[:5]
        raise ValueError(
            "Expected optimizer-update comparison to include only LoRA trainable parameters; "
            f"found non-LoRA parameter names: {sample}"
        )


def _copy_moments(moments: Moments) -> Moments:
    return {name: (exp_avg.clone(), exp_avg_sq.clone()) for name, (exp_avg, exp_avg_sq) in moments.items()}


def _init_moments(names: list[str], gradients: dict[str, Tensor]) -> Moments:
    return {
        name: (
            torch.zeros_like(gradients[name], dtype=torch.float64),
            torch.zeros_like(gradients[name], dtype=torch.float64),
        )
        for name in names
    }


def _global_grad_norm(gradients: dict[str, Tensor], names: list[str]) -> Tensor:
    total = torch.zeros((), dtype=torch.float64)
    for name in names:
        grad = gradients[name].to(torch.float64)
        total += torch.sum(grad * grad)
    return torch.sqrt(total)


def _clip_coef(grad_norm: Tensor, max_norm: float | None) -> Tensor:
    if max_norm is None:
        return torch.ones((), dtype=torch.float64)
    max_norm_tensor = torch.tensor(float(max_norm), dtype=torch.float64)
    return torch.clamp(max_norm_tensor / torch.clamp(grad_norm, min=torch.finfo(torch.float64).eps), max=1.0)


def _adamw_update_delta(
    trainable_state: dict[str, Tensor],
    gradients: dict[str, Tensor],
    moments: Moments,
    names: list[str],
    optimizer_step: int,
    config: AdamWConfig,
) -> tuple[dict[str, Tensor], float, float]:
    grad_norm = _global_grad_norm(gradients, names)
    clip_coef = _clip_coef(grad_norm, config.max_norm)
    bias_correction1 = 1.0 - config.beta1**optimizer_step
    bias_correction2 = 1.0 - config.beta2**optimizer_step
    step_size = config.lr / bias_correction1
    bias_correction2_sqrt = math.sqrt(bias_correction2)

    deltas = {}
    for name in names:
        grad = gradients[name].to(torch.float64) * clip_coef
        param = trainable_state[name].to(torch.float64)
        exp_avg, exp_avg_sq = moments[name]
        exp_avg.mul_(config.beta1).add_(grad, alpha=1.0 - config.beta1)
        exp_avg_sq.mul_(config.beta2).addcmul_(grad, grad, value=1.0 - config.beta2)
        denom = exp_avg_sq.sqrt() / bias_correction2_sqrt
        denom.add_(config.eps)
        adam_delta = torch.div(exp_avg, denom).mul_(-step_size)
        weight_decay_delta = param.mul(-config.lr * config.weight_decay)
        deltas[name] = weight_decay_delta + adam_delta
    return deltas, grad_norm.item(), clip_coef.item()


def _gradient_metrics(reference: dict[str, Tensor], candidate: dict[str, Tensor], names: list[str]) -> dict[str, float]:
    reference_sq = torch.zeros((), dtype=torch.float64)
    candidate_sq = torch.zeros((), dtype=torch.float64)
    diff_sq = torch.zeros((), dtype=torch.float64)
    dot = torch.zeros((), dtype=torch.float64)
    max_abs_diff = torch.zeros((), dtype=torch.float64)
    count = 0
    for name in names:
        ref = reference[name].reshape(-1).to(torch.float64)
        cand = candidate[name].reshape(-1).to(torch.float64)
        diff = cand - ref
        reference_sq += torch.dot(ref, ref)
        candidate_sq += torch.dot(cand, cand)
        diff_sq += torch.dot(diff, diff)
        dot += torch.dot(ref, cand)
        max_abs_diff = torch.maximum(max_abs_diff, diff.abs().max())
        count += ref.numel()

    reference_norm = torch.sqrt(reference_sq)
    candidate_norm = torch.sqrt(candidate_sq)
    diff_norm = torch.sqrt(diff_sq)
    denominator = torch.clamp(reference_norm, min=torch.finfo(torch.float64).eps)
    cosine_denominator = torch.clamp(reference_norm * candidate_norm, min=torch.finfo(torch.float64).eps)
    return {
        "cosine": (dot / cosine_denominator).item(),
        "relative_l2": (diff_norm / denominator).item(),
        "max_abs_diff": max_abs_diff.item(),
        "mse": (diff_sq / count).item(),
        "reference_norm": reference_norm.item(),
        "candidate_norm": candidate_norm.item(),
        "norm_ratio": (candidate_norm / denominator).item(),
    }


def _prefixed_metrics(metrics: dict[str, float], mode: str, label: str) -> dict[str, float]:
    return {
        f"{mode}_cosine_{label}": metrics["cosine"],
        f"{mode}_relative_l2_{label}": metrics["relative_l2"],
        f"{mode}_max_abs_diff_{label}": metrics["max_abs_diff"],
        f"{mode}_mse_{label}": metrics["mse"],
        f"{mode}_candidate_norm_{label}": metrics["candidate_norm"],
        f"{mode}_norm_ratio_{label}": metrics["norm_ratio"],
    }


def _win_counts(candidate_error: Tensor, baseline_error: Tensor, atol: float) -> dict[str, int | float]:
    candidate_wins = (candidate_error < baseline_error - atol).sum().item()
    baseline_wins = (baseline_error < candidate_error - atol).sum().item()
    total = candidate_error.numel()
    ties = total - candidate_wins - baseline_wins
    return {
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "total": total,
        "candidate_win_rate": candidate_wins / total,
        "baseline_win_rate": baseline_wins / total,
        "tie_rate": ties / total,
    }


def _win_metrics(
    reference: dict[str, Tensor],
    candidate: dict[str, Tensor],
    baseline: dict[str, Tensor],
    names: list[str],
    mode: str,
    atol: float,
) -> dict[str, int | float]:
    candidate_element_wins = 0
    baseline_element_wins = 0
    element_ties = 0
    element_count = 0
    candidate_param_errors = []
    baseline_param_errors = []

    for name in names:
        ref = reference[name].reshape(-1).to(torch.float64)
        cand = candidate[name].reshape(-1).to(torch.float64)
        base = baseline[name].reshape(-1).to(torch.float64)
        candidate_error = (cand - ref).abs()
        baseline_error = (base - ref).abs()
        element_counts = _win_counts(candidate_error, baseline_error, atol)
        candidate_element_wins += int(element_counts["candidate_wins"])
        baseline_element_wins += int(element_counts["baseline_wins"])
        element_ties += int(element_counts["ties"])
        element_count += int(element_counts["total"])

        denominator = torch.clamp(torch.linalg.vector_norm(ref), min=torch.finfo(torch.float64).eps)
        candidate_param_errors.append(torch.linalg.vector_norm(cand - ref) / denominator)
        baseline_param_errors.append(torch.linalg.vector_norm(base - ref) / denominator)

    param_counts = _win_counts(torch.stack(candidate_param_errors), torch.stack(baseline_param_errors), atol)
    return {
        f"{mode}_candidate_element_win_count": candidate_element_wins,
        f"{mode}_baseline_element_win_count": baseline_element_wins,
        f"{mode}_element_tie_count": element_ties,
        f"{mode}_element_count": element_count,
        f"{mode}_candidate_element_win_rate": candidate_element_wins / element_count,
        f"{mode}_baseline_element_win_rate": baseline_element_wins / element_count,
        f"{mode}_element_tie_rate": element_ties / element_count,
        f"{mode}_candidate_param_win_count": param_counts["candidate_wins"],
        f"{mode}_baseline_param_win_count": param_counts["baseline_wins"],
        f"{mode}_param_tie_count": param_counts["ties"],
        f"{mode}_param_count": param_counts["total"],
        f"{mode}_candidate_param_win_rate": param_counts["candidate_win_rate"],
        f"{mode}_baseline_param_win_rate": param_counts["baseline_win_rate"],
        f"{mode}_param_tie_rate": param_counts["tie_rate"],
    }


def _mode_metrics(
    reference_delta: dict[str, Tensor],
    candidate_delta: dict[str, Tensor],
    baseline_delta: dict[str, Tensor],
    names: list[str],
    mode: str,
    candidate_label: str,
    baseline_label: str,
    win_tie_atol: float,
) -> dict[str, Any]:
    candidate_metrics = _gradient_metrics(reference_delta, candidate_delta, names)
    baseline_metrics = _gradient_metrics(reference_delta, baseline_delta, names)
    return {
        f"{mode}_reference_update_norm": candidate_metrics["reference_norm"],
        **_prefixed_metrics(candidate_metrics, mode, candidate_label),
        **_prefixed_metrics(baseline_metrics, mode, baseline_label),
        **_win_metrics(reference_delta, candidate_delta, baseline_delta, names, mode, win_tie_atol),
    }


def _write_json(path: Path | None, results: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)


def _print_header(candidate_label: str, baseline_label: str) -> None:
    print(
        "step compared "
        f"shared_state_relative_l2_{baseline_label} shared_state_relative_l2_{candidate_label} "
        f"method_state_relative_l2_{baseline_label} method_state_relative_l2_{candidate_label} "
        "shared_state_candidate_param_win_rate method_state_candidate_param_win_rate",
        flush=True,
    )


def _print_result(result: dict[str, Any], candidate_label: str, baseline_label: str) -> None:
    print(
        f"{result['step']} {result['compared']} "
        f"{result[f'shared_state_relative_l2_{baseline_label}']:.8e} "
        f"{result[f'shared_state_relative_l2_{candidate_label}']:.8e} "
        f"{result[f'method_state_relative_l2_{baseline_label}']:.8e} "
        f"{result[f'method_state_relative_l2_{candidate_label}']:.8e} "
        f"{result['shared_state_candidate_param_win_rate']:.8f} "
        f"{result['method_state_candidate_param_win_rate']:.8f}",
        flush=True,
    )


def compare_optimizer_updates(
    reference_dir: Path,
    candidate_dir: Path,
    baseline_dir: Path,
    candidate_label: str,
    baseline_label: str,
    config: AdamWConfig,
    win_tie_atol: float,
    require_lora_params: bool,
    json_output: Path | None,
) -> list[dict[str, Any]]:
    common_steps = _common_steps(reference_dir, candidate_dir, baseline_dir)
    results: list[dict[str, Any]] = []
    ref_moments: Moments | None = None
    candidate_moments: Moments | None = None
    baseline_moments: Moments | None = None

    _print_header(candidate_label, baseline_label)
    with torch.inference_mode():
        for optimizer_step, step in enumerate(common_steps, start=1):
            reference_gradients, reference_metadata = _load_step(reference_dir, step)
            candidate_gradients, candidate_metadata = _load_step(candidate_dir, step)
            baseline_gradients, baseline_metadata = _load_step(baseline_dir, step)
            trainable_state = _load_reference_trainable_state(reference_dir, step)

            names = _validate_tensor_dict(reference_gradients, candidate_gradients, candidate_label)
            _validate_tensor_dict(reference_gradients, baseline_gradients, baseline_label)
            _validate_tensor_dict(reference_gradients, trainable_state, "reference trainable state")
            if require_lora_params:
                _validate_lora_names(names)

            if ref_moments is None:
                ref_moments = _init_moments(names, reference_gradients)
                candidate_moments = _init_moments(names, reference_gradients)
                baseline_moments = _init_moments(names, reference_gradients)
            assert candidate_moments is not None
            assert baseline_moments is not None

            shared_reference_delta, shared_reference_grad_norm, shared_reference_clip_coef = _adamw_update_delta(
                trainable_state,
                reference_gradients,
                _copy_moments(ref_moments),
                names,
                optimizer_step,
                config,
            )
            shared_candidate_delta, shared_candidate_grad_norm, shared_candidate_clip_coef = _adamw_update_delta(
                trainable_state,
                candidate_gradients,
                _copy_moments(ref_moments),
                names,
                optimizer_step,
                config,
            )
            shared_baseline_delta, shared_baseline_grad_norm, shared_baseline_clip_coef = _adamw_update_delta(
                trainable_state,
                baseline_gradients,
                _copy_moments(ref_moments),
                names,
                optimizer_step,
                config,
            )

            method_reference_delta, method_reference_grad_norm, method_reference_clip_coef = _adamw_update_delta(
                trainable_state,
                reference_gradients,
                ref_moments,
                names,
                optimizer_step,
                config,
            )
            method_candidate_delta, method_candidate_grad_norm, method_candidate_clip_coef = _adamw_update_delta(
                trainable_state,
                candidate_gradients,
                candidate_moments,
                names,
                optimizer_step,
                config,
            )
            method_baseline_delta, method_baseline_grad_norm, method_baseline_clip_coef = _adamw_update_delta(
                trainable_state,
                baseline_gradients,
                baseline_moments,
                names,
                optimizer_step,
                config,
            )

            candidate_batch_hash_match = reference_metadata.get("batch_hash") == candidate_metadata.get("batch_hash")
            baseline_batch_hash_match = reference_metadata.get("batch_hash") == baseline_metadata.get("batch_hash")
            candidate_state_hash_match = reference_metadata.get("pre_step_state_hash") == candidate_metadata.get(
                "pre_step_state_hash"
            )
            baseline_state_hash_match = reference_metadata.get("pre_step_state_hash") == baseline_metadata.get(
                "pre_step_state_hash"
            )
            result = {
                "step": step,
                "optimizer_step": optimizer_step,
                "compared": (
                    candidate_batch_hash_match
                    and baseline_batch_hash_match
                    and candidate_state_hash_match
                    and baseline_state_hash_match
                ),
                "candidate_batch_hash_match": candidate_batch_hash_match,
                "baseline_batch_hash_match": baseline_batch_hash_match,
                "candidate_pre_step_state_hash_match": candidate_state_hash_match,
                "baseline_pre_step_state_hash_match": baseline_state_hash_match,
                "parameter_count": len(names),
                "optimizer": {
                    "type": "adamw",
                    "lr": config.lr,
                    "beta1": config.beta1,
                    "beta2": config.beta2,
                    "eps": config.eps,
                    "weight_decay": config.weight_decay,
                    "max_norm": config.max_norm,
                },
                "shared_state_reference_grad_norm": shared_reference_grad_norm,
                f"shared_state_{candidate_label}_grad_norm": shared_candidate_grad_norm,
                f"shared_state_{baseline_label}_grad_norm": shared_baseline_grad_norm,
                "shared_state_reference_clip_coef": shared_reference_clip_coef,
                f"shared_state_{candidate_label}_clip_coef": shared_candidate_clip_coef,
                f"shared_state_{baseline_label}_clip_coef": shared_baseline_clip_coef,
                "method_state_reference_grad_norm": method_reference_grad_norm,
                f"method_state_{candidate_label}_grad_norm": method_candidate_grad_norm,
                f"method_state_{baseline_label}_grad_norm": method_baseline_grad_norm,
                "method_state_reference_clip_coef": method_reference_clip_coef,
                f"method_state_{candidate_label}_clip_coef": method_candidate_clip_coef,
                f"method_state_{baseline_label}_clip_coef": method_baseline_clip_coef,
                **_mode_metrics(
                    shared_reference_delta,
                    shared_candidate_delta,
                    shared_baseline_delta,
                    names,
                    "shared_state",
                    candidate_label,
                    baseline_label,
                    win_tie_atol,
                ),
                **_mode_metrics(
                    method_reference_delta,
                    method_candidate_delta,
                    method_baseline_delta,
                    names,
                    "method_state",
                    candidate_label,
                    baseline_label,
                    win_tie_atol,
                ),
            }
            results.append(result)
            _print_result(result, candidate_label, baseline_label)
            _write_json(json_output, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AdamW optimizer update deltas implied by gradient diagnostic artifacts."
    )
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--baseline-candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--win-tie-atol", type=float, default=0.0)
    parser.add_argument(
        "--allow-non-lora-params",
        action="store_true",
        help="Allow comparison of all saved trainable params instead of requiring LoRA parameter names.",
    )
    args = parser.parse_args()

    if args.win_tie_atol < 0:
        raise ValueError("--win-tie-atol must be non-negative")
    if args.candidate_label == args.baseline_label:
        raise ValueError("--candidate-label and --baseline-label must be distinct")
    if args.max_norm is not None and args.max_norm < 0:
        raise ValueError("--max-norm must be non-negative")

    compare_optimizer_updates(
        reference_dir=args.reference_dir,
        candidate_dir=args.candidate_dir,
        baseline_dir=args.baseline_candidate_dir,
        candidate_label=args.candidate_label,
        baseline_label=args.baseline_label,
        config=AdamWConfig(
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
            max_norm=args.max_norm,
        ),
        win_tie_atol=args.win_tie_atol,
        require_lora_params=not args.allow_non_lora_params,
        json_output=args.json_output,
    )


if __name__ == "__main__":
    main()
