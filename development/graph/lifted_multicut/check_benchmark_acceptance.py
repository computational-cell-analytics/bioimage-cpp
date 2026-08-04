from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


PROBLEMS = ("2d", "3d", "grid")


def load_results(
    path: Path,
    *,
    latest_run_only: bool,
) -> dict[tuple[str, str], dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row.get("record_type") == "result":
                rows.append(row)
    if latest_run_only and rows:
        latest_run_id = rows[-1]["run_id"]
        rows = [row for row in rows if row["run_id"] == latest_run_id]
    return {(row["problem"], row["solver"]): row for row in rows}


def check_raw_measurements(
    row: dict,
    backend: str,
    expected_repeats: int,
) -> list[str]:
    errors = []
    runtimes = row[f"{backend}_runtimes_s"]
    energies = row[f"{backend}_energies"]
    digests = row[f"{backend}_label_digests"]
    if len(runtimes) != expected_repeats:
        errors.append(
            f"{backend} has {len(runtimes)} runtimes; expected {expected_repeats}"
        )
    if len(energies) != expected_repeats:
        errors.append(
            f"{backend} has {len(energies)} energies; expected {expected_repeats}"
        )
    if len(digests) != expected_repeats:
        errors.append(
            f"{backend} has {len(digests)} digests; expected {expected_repeats}"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in runtimes):
        errors.append(f"{backend} runtimes must be finite and positive")
    if any(not math.isfinite(value) for value in energies):
        errors.append(f"{backend} energies must be finite")
    if energies and len(set(energies)) != 1:
        errors.append(f"{backend} energies differ between repeats")
    if digests and len(set(digests)) != 1:
        errors.append(f"{backend} label digests differ between repeats")
    if runtimes:
        if median(runtimes) != row[f"{backend}_runtime_median_s"]:
            errors.append(f"{backend} runtime median does not match raw runtimes")
        if min(runtimes) != row[f"{backend}_runtime_min_s"]:
            errors.append(f"{backend} runtime minimum does not match raw runtimes")
    if energies and median(energies) != row[f"{backend}_energy"]:
        errors.append(f"{backend} energy does not match raw energies")
    return errors


def check_problem(
    baseline: dict,
    candidate: dict,
    problem: str,
    solver: str,
    expected_repeats: int,
    max_regression: float,
    min_improvement: float,
    require_improvement: bool,
    require_reference_parity: bool,
) -> tuple[dict, list[str]]:
    key = (problem, solver)
    errors = []
    if key not in baseline:
        return {}, [f"baseline has no row for {problem} / {solver}"]
    if key not in candidate:
        return {}, [f"candidate has no row for {problem} / {solver}"]

    baseline_row = baseline[key]
    candidate_row = candidate[key]
    errors.extend(
        f"baseline {error}"
        for error in check_raw_measurements(
            baseline_row, "bic", expected_repeats
        )
    )
    errors.extend(
        f"candidate {error}"
        for error in check_raw_measurements(
            candidate_row, "bic", expected_repeats
        )
    )
    errors.extend(
        f"candidate {error}"
        for error in check_raw_measurements(
            candidate_row, "nifty", expected_repeats
        )
    )

    if candidate_row["bic_energies"] != baseline_row["bic_energies"]:
        errors.append("candidate energies differ from the baseline")
    if candidate_row["bic_label_digests"] != baseline_row["bic_label_digests"]:
        errors.append("candidate label digests differ from the baseline")

    baseline_runtime = baseline_row["bic_runtime_median_s"]
    candidate_runtime = candidate_row["bic_runtime_median_s"]
    reference_runtime = candidate_row["nifty_runtime_median_s"]
    baseline_ratio = candidate_runtime / baseline_runtime
    reference_ratio = reference_runtime / candidate_runtime
    if baseline_ratio > 1.0 + max_regression:
        errors.append(
            f"runtime ratio {baseline_ratio:.6f} exceeds "
            f"{1.0 + max_regression:.6f}"
        )
    if require_improvement and baseline_ratio > 1.0 - min_improvement:
        errors.append(
            f"runtime ratio {baseline_ratio:.6f} exceeds improvement target "
            f"{1.0 - min_improvement:.6f}"
        )
    if require_reference_parity and candidate_runtime > reference_runtime:
        errors.append(
            f"candidate runtime {candidate_runtime:.6f}s exceeds "
            f"reference runtime {reference_runtime:.6f}s"
        )

    return {
        "problem": problem,
        "baseline_s": baseline_runtime,
        "candidate_s": candidate_runtime,
        "candidate_over_baseline": baseline_ratio,
        "reference_s": reference_runtime,
        "reference_over_candidate": reference_ratio,
        "status": "PASS" if not errors else "FAIL",
    }, errors


def print_table(rows: list[dict]) -> None:
    print(
        "| problem | baseline_s | candidate_s | candidate / baseline | "
        "nifty_s | nifty / candidate | status |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        print(
            f"| {row['problem']} | {row['baseline_s']:.6f} | "
            f"{row['candidate_s']:.6f} | "
            f"{row['candidate_over_baseline']:.6f} | "
            f"{row['reference_s']:.6f} | "
            f"{row['reference_over_candidate']:.6f} | {row['status']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check lifted-KL benchmark results against a saved baseline."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--solver", default="lifted_kernighan_lin")
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=PROBLEMS,
        default=("2d", "3d"),
    )
    parser.add_argument("--expected-repeats", type=int, default=10)
    parser.add_argument("--max-regression", type=float, default=0.03)
    parser.add_argument("--min-improvement", type=float, default=0.03)
    parser.add_argument(
        "--improvement-problems",
        nargs="*",
        choices=PROBLEMS,
        default=("3d",),
        metavar="PROBLEM",
    )
    parser.add_argument(
        "--reference-parity",
        nargs="*",
        choices=PROBLEMS,
        default=("3d",),
        metavar="PROBLEM",
    )
    args = parser.parse_args()

    if args.expected_repeats < 1:
        raise ValueError("--expected-repeats must be at least 1")
    if args.max_regression < 0.0:
        raise ValueError("--max-regression must be non-negative")
    if not 0.0 <= args.min_improvement < 1.0:
        raise ValueError("--min-improvement must be in [0, 1)")

    baseline = load_results(args.baseline, latest_run_only=False)
    candidate = load_results(args.candidate, latest_run_only=True)
    rows = []
    failures = []
    improvement_problems = set(args.improvement_problems)
    parity_problems = set(args.reference_parity)
    for problem in args.problems:
        row, errors = check_problem(
            baseline,
            candidate,
            problem,
            args.solver,
            args.expected_repeats,
            args.max_regression,
            args.min_improvement,
            problem in improvement_problems,
            problem in parity_problems,
        )
        if row:
            rows.append(row)
        failures.extend(
            f"{problem} / {args.solver}: {error}" for error in errors
        )

    print_table(rows)
    if failures:
        raise SystemExit("Acceptance failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
