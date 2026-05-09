import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_METRICS = [
    "compared",
    "batch_hash_match",
    "pre_step_state_hash_match",
    "cosine",
    "relative_l2",
    "max_abs_diff",
    "norm_ratio",
    "candidate_loss_scale_std",
    "candidate_loss_scale_min",
    "candidate_loss_scale_max",
    "candidate_replay_padding_micro_batches",
    "candidate_replay_padding_tokens",
]

DELTA_METRICS = ["cosine", "relative_l2", "max_abs_diff", "norm_ratio"]


def _load_results(path: Path) -> dict[int, dict[str, Any]]:
    with open(path) as f:
        results = json.load(f)

    by_step = {}
    for result in results:
        step = result["step"]
        if step in by_step:
            raise ValueError(f"Duplicate step {step} in {path}")
        by_step[step] = result
    return by_step


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8e}"
    return str(value)


def _combined_rows(
    existing_results: dict[int, dict[str, Any]],
    fix_results: dict[int, dict[str, Any]],
    existing_label: str,
    fix_label: str,
    metrics: list[str],
    include_deltas: bool,
) -> list[dict[str, Any]]:
    common_steps = sorted(set(existing_results) & set(fix_results))
    if not common_steps:
        raise ValueError("No common steps found between comparison files")

    rows = []
    for step in common_steps:
        existing = existing_results[step]
        fix = fix_results[step]
        row: dict[str, Any] = {"step": step}
        for metric in metrics:
            row[f"{metric}_{existing_label}"] = existing.get(metric)
            row[f"{metric}_{fix_label}"] = fix.get(metric)
        if include_deltas:
            for metric in DELTA_METRICS:
                existing_value = existing.get(metric)
                fix_value = fix.get(metric)
                if existing_value is None or fix_value is None:
                    row[f"{metric}_delta_{fix_label}_minus_{existing_label}"] = None
                else:
                    row[f"{metric}_delta_{fix_label}_minus_{existing_label}"] = fix_value - existing_value
        rows.append(row)
    return rows


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def _write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        _write_json(path, rows)
    elif path.suffix == ".csv":
        _write_csv(path, rows)
    elif path.suffix in {".md", ".markdown"}:
        _write_markdown(path, rows)
    else:
        raise ValueError("Output path must end in .json, .csv, .md, or .markdown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize two gradient diagnostic comparison JSON files.")
    parser.add_argument("existing_json", type=Path, help="Comparison JSON for the existing gradient method.")
    parser.add_argument("fix_json", type=Path, help="Comparison JSON for the fixed gradient method.")
    parser.add_argument("--output", required=True, type=Path, help="Combined output path: .json, .csv, or .md.")
    parser.add_argument("--existing-label", default="existing_method")
    parser.add_argument("--fix-label", default="fix")
    parser.add_argument("--no-deltas", action="store_true", help="Do not include fix-minus-existing delta columns.")
    args = parser.parse_args()

    existing_results = _load_results(args.existing_json)
    fix_results = _load_results(args.fix_json)
    rows = _combined_rows(
        existing_results=existing_results,
        fix_results=fix_results,
        existing_label=args.existing_label,
        fix_label=args.fix_label,
        metrics=DEFAULT_METRICS,
        include_deltas=not args.no_deltas,
    )
    _write_output(args.output, rows)
    print(f"Wrote {len(rows)} combined steps to {args.output}")


if __name__ == "__main__":
    main()
