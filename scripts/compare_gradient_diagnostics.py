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


def _compare_step(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    reference_gradients, reference_metadata = _load_step(reference_dir)
    candidate_gradients, candidate_metadata = _load_step(candidate_dir)

    reference_names = list(reference_gradients)
    candidate_names = list(candidate_gradients)
    if reference_names != candidate_names:
        missing = sorted(set(reference_names) - set(candidate_names))
        extra = sorted(set(candidate_names) - set(reference_names))
        raise ValueError(f"Gradient parameter names differ: missing={missing}, extra={extra}")

    for name in reference_names:
        if reference_gradients[name].shape != candidate_gradients[name].shape:
            raise ValueError(
                f"Gradient shape differs for {name}: "
                f"{tuple(reference_gradients[name].shape)} != {tuple(candidate_gradients[name].shape)}"
            )

    reference_flat = _flatten(reference_gradients, reference_names)
    candidate_flat = _flatten(candidate_gradients, reference_names)
    diff = candidate_flat - reference_flat
    reference_norm = torch.linalg.vector_norm(reference_flat)
    candidate_norm = torch.linalg.vector_norm(candidate_flat)
    diff_norm = torch.linalg.vector_norm(diff)
    denominator = torch.clamp(reference_norm, min=torch.finfo(torch.float64).eps)

    batch_hash_match = reference_metadata.get("batch_hash") == candidate_metadata.get("batch_hash")
    state_hash_match = reference_metadata.get("pre_step_state_hash") == candidate_metadata.get("pre_step_state_hash")

    return {
        "step": reference_metadata["step"],
        "compared": batch_hash_match and state_hash_match,
        "batch_hash_match": batch_hash_match,
        "pre_step_state_hash_match": state_hash_match,
        "cosine": torch.nn.functional.cosine_similarity(reference_flat, candidate_flat, dim=0).item(),
        "relative_l2": (diff_norm / denominator).item(),
        "max_abs_diff": diff.abs().max().item(),
        "reference_norm": reference_norm.item(),
        "candidate_norm": candidate_norm.item(),
        "norm_ratio": (candidate_norm / denominator).item(),
        "candidate_loss_scale_std": candidate_metadata.get("dp_loss_scale_std"),
        "candidate_loss_scale_min": candidate_metadata.get("dp_loss_scale_min"),
        "candidate_loss_scale_max": candidate_metadata.get("dp_loss_scale_max"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recorded RL gradient diagnostic artifacts.")
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    reference_steps = _step_dirs(args.reference_dir)
    candidate_steps = _step_dirs(args.candidate_dir)
    common_steps = sorted(set(reference_steps) & set(candidate_steps))
    if not common_steps:
        raise ValueError("No common gradient diagnostic steps found")

    results = [_compare_step(reference_steps[step], candidate_steps[step]) for step in common_steps]

    print("step compared cosine relative_l2 max_abs_diff norm_ratio loss_scale_std")
    for result in results:
        print(
            "{step} {compared} {cosine:.8f} {relative_l2:.8e} "
            "{max_abs_diff:.8e} {norm_ratio:.8f} {candidate_loss_scale_std}".format(**result)
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
