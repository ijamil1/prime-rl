import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _load_step(path: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    gradients = torch.load(path / "gradients.pt", map_location="cpu", weights_only=False)
    with open(path / "metadata.json") as f:
        metadata = json.load(f)
    return gradients, metadata


def _step_dirs(root: Path) -> dict[int, Path]:
    result = {}
    for path in root.glob("step_*"):
        if path.is_dir():
            result[int(path.name.removeprefix("step_"))] = path
    return result


def _flatten(gradients: dict[str, Tensor], names: list[str]) -> Tensor:
    return torch.cat([gradients[name].reshape(-1).to(torch.float64) for name in names])


def _validate_gradients(
    reference_gradients: dict[str, Tensor], candidate_gradients: dict[str, Tensor], label: str = "candidate"
) -> list[str]:
    reference_names = list(reference_gradients)
    candidate_names = list(candidate_gradients)
    if reference_names != candidate_names:
        missing = sorted(set(reference_names) - set(candidate_names))
        extra = sorted(set(candidate_names) - set(reference_names))
        raise ValueError(f"Gradient parameter names differ for {label}: missing={missing}, extra={extra}")

    for name in reference_names:
        if reference_gradients[name].shape != candidate_gradients[name].shape:
            raise ValueError(
                f"Gradient shape differs for {label} parameter {name}: "
                f"{tuple(reference_gradients[name].shape)} != {tuple(candidate_gradients[name].shape)}"
            )
    return reference_names


def _gradient_metrics(reference_flat: Tensor, candidate_flat: Tensor) -> dict[str, float]:
    diff = candidate_flat - reference_flat
    reference_norm = torch.linalg.vector_norm(reference_flat)
    candidate_norm = torch.linalg.vector_norm(candidate_flat)
    diff_norm = torch.linalg.vector_norm(diff)
    denominator = torch.clamp(reference_norm, min=torch.finfo(torch.float64).eps)
    return {
        "cosine": torch.nn.functional.cosine_similarity(reference_flat, candidate_flat, dim=0).item(),
        "relative_l2": (diff_norm / denominator).item(),
        "max_abs_diff": diff.abs().max().item(),
        "reference_norm": reference_norm.item(),
        "candidate_norm": candidate_norm.item(),
        "norm_ratio": (candidate_norm / denominator).item(),
    }


def _prefixed_metrics(metrics: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        f"cosine_{label}": metrics["cosine"],
        f"relative_l2_{label}": metrics["relative_l2"],
        f"max_abs_diff_{label}": metrics["max_abs_diff"],
        f"candidate_norm_{label}": metrics["candidate_norm"],
        f"norm_ratio_{label}": metrics["norm_ratio"],
    }


def _prefixed_candidate_metadata(metadata: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        f"{label}_loss_scale_std": metadata.get("dp_loss_scale_std"),
        f"{label}_loss_scale_min": metadata.get("dp_loss_scale_min"),
        f"{label}_loss_scale_max": metadata.get("dp_loss_scale_max"),
        f"{label}_replay_padding_micro_batches": metadata.get("replay_padding_micro_batches"),
        f"{label}_replay_padding_tokens": metadata.get("replay_padding_tokens"),
        f"{label}_execution_batch_hash": metadata.get("execution_batch_hash"),
        f"{label}_execution_local_batch_hash": metadata.get("execution_local_batch_hash"),
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


def _element_win_metrics(
    reference_flat: Tensor, candidate_flat: Tensor, baseline_flat: Tensor, atol: float
) -> dict[str, int | float]:
    counts = _win_counts((candidate_flat - reference_flat).abs(), (baseline_flat - reference_flat).abs(), atol)
    return {
        "candidate_element_win_count": counts["candidate_wins"],
        "baseline_element_win_count": counts["baseline_wins"],
        "element_tie_count": counts["ties"],
        "element_count": counts["total"],
        "candidate_element_win_rate": counts["candidate_win_rate"],
        "baseline_element_win_rate": counts["baseline_win_rate"],
        "element_tie_rate": counts["tie_rate"],
    }


def _parameter_win_metrics(
    reference_gradients: dict[str, Tensor],
    candidate_gradients: dict[str, Tensor],
    baseline_gradients: dict[str, Tensor],
    names: list[str],
    atol: float,
) -> dict[str, int | float]:
    candidate_errors = []
    baseline_errors = []
    for name in names:
        reference = reference_gradients[name].reshape(-1).to(torch.float64)
        candidate = candidate_gradients[name].reshape(-1).to(torch.float64)
        baseline = baseline_gradients[name].reshape(-1).to(torch.float64)
        denominator = torch.clamp(torch.linalg.vector_norm(reference), min=torch.finfo(torch.float64).eps)
        candidate_errors.append(torch.linalg.vector_norm(candidate - reference) / denominator)
        baseline_errors.append(torch.linalg.vector_norm(baseline - reference) / denominator)

    counts = _win_counts(torch.stack(candidate_errors), torch.stack(baseline_errors), atol)
    return {
        "candidate_param_win_count": counts["candidate_wins"],
        "baseline_param_win_count": counts["baseline_wins"],
        "param_tie_count": counts["ties"],
        "param_count": counts["total"],
        "candidate_param_win_rate": counts["candidate_win_rate"],
        "baseline_param_win_rate": counts["baseline_win_rate"],
        "param_tie_rate": counts["tie_rate"],
    }


def _compare_step(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    reference_gradients, reference_metadata = _load_step(reference_dir)
    candidate_gradients, candidate_metadata = _load_step(candidate_dir)
    reference_names = _validate_gradients(reference_gradients, candidate_gradients)

    reference_flat = _flatten(reference_gradients, reference_names)
    candidate_flat = _flatten(candidate_gradients, reference_names)
    metrics = _gradient_metrics(reference_flat, candidate_flat)

    batch_hash_match = reference_metadata.get("batch_hash") == candidate_metadata.get("batch_hash")
    state_hash_match = reference_metadata.get("pre_step_state_hash") == candidate_metadata.get("pre_step_state_hash")

    return {
        "step": reference_metadata["step"],
        "compared": batch_hash_match and state_hash_match,
        "batch_hash_match": batch_hash_match,
        "pre_step_state_hash_match": state_hash_match,
        **metrics,
        "candidate_loss_scale_std": candidate_metadata.get("dp_loss_scale_std"),
        "candidate_loss_scale_min": candidate_metadata.get("dp_loss_scale_min"),
        "candidate_loss_scale_max": candidate_metadata.get("dp_loss_scale_max"),
    }


def _compare_step_three_way(
    reference_dir: Path,
    candidate_dir: Path,
    baseline_dir: Path,
    candidate_label: str,
    baseline_label: str,
    win_tie_atol: float,
) -> dict[str, Any]:
    reference_gradients, reference_metadata = _load_step(reference_dir)
    candidate_gradients, candidate_metadata = _load_step(candidate_dir)
    baseline_gradients, baseline_metadata = _load_step(baseline_dir)

    reference_names = _validate_gradients(reference_gradients, candidate_gradients, candidate_label)
    _validate_gradients(reference_gradients, baseline_gradients, baseline_label)

    reference_flat = _flatten(reference_gradients, reference_names)
    candidate_flat = _flatten(candidate_gradients, reference_names)
    baseline_flat = _flatten(baseline_gradients, reference_names)
    candidate_metrics = _gradient_metrics(reference_flat, candidate_flat)
    baseline_metrics = _gradient_metrics(reference_flat, baseline_flat)

    candidate_batch_hash_match = reference_metadata.get("batch_hash") == candidate_metadata.get("batch_hash")
    baseline_batch_hash_match = reference_metadata.get("batch_hash") == baseline_metadata.get("batch_hash")
    candidate_state_hash_match = reference_metadata.get("pre_step_state_hash") == candidate_metadata.get(
        "pre_step_state_hash"
    )
    baseline_state_hash_match = reference_metadata.get("pre_step_state_hash") == baseline_metadata.get(
        "pre_step_state_hash"
    )

    return {
        "step": reference_metadata["step"],
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
        "reference_norm": candidate_metrics["reference_norm"],
        **_prefixed_metrics(candidate_metrics, candidate_label),
        **_prefixed_metrics(baseline_metrics, baseline_label),
        **_prefixed_candidate_metadata(candidate_metadata, candidate_label),
        **_prefixed_candidate_metadata(baseline_metadata, baseline_label),
        **_element_win_metrics(reference_flat, candidate_flat, baseline_flat, win_tie_atol),
        **_parameter_win_metrics(
            reference_gradients, candidate_gradients, baseline_gradients, reference_names, win_tie_atol
        ),
    }


def _common_steps(*roots: Path) -> list[int]:
    step_sets = [set(_step_dirs(root)) for root in roots]
    common_steps = sorted(set.intersection(*step_sets))
    if not common_steps:
        raise ValueError("No common gradient diagnostic steps found")
    return common_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recorded RL gradient diagnostic artifacts.")
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--baseline-candidate-dir", type=Path)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--win-tie-atol", type=float, default=0.0)
    args = parser.parse_args()

    if args.win_tie_atol < 0:
        raise ValueError("--win-tie-atol must be non-negative")
    if args.baseline_candidate_dir is not None and args.candidate_label == args.baseline_label:
        raise ValueError("--candidate-label and --baseline-label must be distinct")

    reference_steps = _step_dirs(args.reference_dir)
    candidate_steps = _step_dirs(args.candidate_dir)
    if args.baseline_candidate_dir is None:
        common_steps = _common_steps(args.reference_dir, args.candidate_dir)
        results = [_compare_step(reference_steps[step], candidate_steps[step]) for step in common_steps]

        print("step compared cosine relative_l2 max_abs_diff norm_ratio loss_scale_std")
        for result in results:
            print(
                "{step} {compared} {cosine:.8f} {relative_l2:.8e} "
                "{max_abs_diff:.8e} {norm_ratio:.8f} {candidate_loss_scale_std}".format(**result)
            )
    else:
        baseline_steps = _step_dirs(args.baseline_candidate_dir)
        common_steps = _common_steps(args.reference_dir, args.candidate_dir, args.baseline_candidate_dir)
        results = [
            _compare_step_three_way(
                reference_steps[step],
                candidate_steps[step],
                baseline_steps[step],
                args.candidate_label,
                args.baseline_label,
                args.win_tie_atol,
            )
            for step in common_steps
        ]

        cosine_baseline = f"cosine_{args.baseline_label}"
        cosine_candidate = f"cosine_{args.candidate_label}"
        relative_l2_baseline = f"relative_l2_{args.baseline_label}"
        relative_l2_candidate = f"relative_l2_{args.candidate_label}"
        print(
            "step compared "
            f"{cosine_baseline} {cosine_candidate} "
            f"{relative_l2_baseline} {relative_l2_candidate} "
            "candidate_element_win_rate candidate_param_win_rate"
        )
        for result in results:
            print(
                f"{result['step']} {result['compared']} "
                f"{result[cosine_baseline]:.8f} {result[cosine_candidate]:.8f} "
                f"{result[relative_l2_baseline]:.8e} {result[relative_l2_candidate]:.8e} "
                f"{result['candidate_element_win_rate']:.8f} {result['candidate_param_win_rate']:.8f}"
            )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
