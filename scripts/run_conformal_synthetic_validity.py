"""Phase 1 validity study: the conformal core on synthetic scores with known laws.

Runs exactly the cases the tests run (same seeds, via `poseconf.synthetic_validity`) and writes
`results/validity/conformal_core_synthetic.json`, so Phase 1's numbers come from a committed file.
CPU only, no dataset, no P1. Exits non-zero if any case fails its pre-registered check.

    python scripts/run_conformal_synthetic_validity.py --out results/validity/conformal_core_synthetic.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from poseconf.conformal.metrics import mc_standard_error
from poseconf.provenance import RESULT_SCHEMA, poseconf_git_sha, utc_now_iso, write_result_json
from poseconf.synthetic_validity import (
    DISTRIBUTIONS,
    SEED,
    case_rng,
    covariate_shift_draws,
    split_case_key,
    split_coverage_draws,
    summarise_split_draws,
)

#: Study design — identical to tests/test_conformal_coverage_law.py and test_conformal_weighted.py.
N_RESAMPLES = 2000
N_CAL = (100, 1000)
ALPHAS = (0.05, 0.10)
BAND_LEVEL = 0.99
KS_MIN_P = 0.01
FAILURE_CASE = {"dist": "gaussian", "n": 1000, "alpha": 0.10, "failure_rate": 0.05}
SHIFT_CASE = {"n": 500, "n_test": 200, "alphas": (0.10, 0.20)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=Path("results/validity/conformal_core_synthetic.json")
    )
    return parser.parse_args(argv)


def _split_cases() -> list[dict[str, Any]]:
    min_frac = float(stats.binom.ppf(0.001, N_RESAMPLES, BAND_LEVEL) / N_RESAMPLES)
    cases = []
    for dist in DISTRIBUTIONS:
        for n in N_CAL:
            for alpha in ALPHAS:
                draws = split_coverage_draws(
                    dist, n, alpha, N_RESAMPLES, case_rng(*split_case_key(dist, n, alpha))
                )
                s = summarise_split_draws(draws, n, alpha, level=BAND_LEVEL)
                lo, hi = s["theory_interval"]
                s["mean_in_theory_interval_3se"] = bool(
                    lo - 3 * s["mc_se"] <= s["mean_coverage"] <= hi + 3 * s["mc_se"]
                )
                s["verdict"] = (
                    "VALID"
                    if s["ks_pvalue"] > KS_MIN_P
                    and s["mean_in_band"]
                    and s["frac_draws_in_band"] >= min_frac
                    and s["mean_in_theory_interval_3se"]
                    else "DEVIATES"
                )
                cases.append({"distribution": dist, "convention": "none (no failures)", **s})
    return cases


def _failure_cases() -> list[dict[str, Any]]:
    out = []
    for i, convention in enumerate(("answer_required", "abstain_allowed")):
        c = FAILURE_CASE
        draws = split_coverage_draws(
            c["dist"],
            c["n"],
            c["alpha"],
            N_RESAMPLES,
            case_rng(7, c["n"], i),
            failure_rate=c["failure_rate"],
            convention=convention,
        )
        s = summarise_split_draws(draws, c["n"], c["alpha"], level=BAND_LEVEL)
        z = float(stats.norm.ppf(1 - (1 - BAND_LEVEL) / 2))
        valid = s["mean_coverage"] >= 1 - c["alpha"] - z * s["mc_se"]
        out.append(
            {
                "distribution": c["dist"],
                "convention": convention,
                "failure_rate": c["failure_rate"],
                "n_cal": c["n"],
                "alpha": c["alpha"],
                "n_resamples": N_RESAMPLES,
                "mean_coverage": s["mean_coverage"],
                "mc_se": s["mc_se"],
                "check": "mean >= 1 - alpha - z99 * se (atoms at +/-inf: law is conservative)",
                "verdict": "VALID" if valid else "DEVIATES",
            }
        )
    return out


def _shift_cases() -> list[dict[str, Any]]:
    out = []
    for alpha in SHIFT_CASE["alphas"]:
        key = round(alpha * 100)
        res = {}
        for name, seed, weighted in (("weighted", 21, True), ("split", 22, False)):
            draws = covariate_shift_draws(
                SHIFT_CASE["n"],
                alpha,
                N_RESAMPLES,
                SHIFT_CASE["n_test"],
                case_rng(seed, key),
                weighted=weighted,
            )
            res[name] = {
                "mean_coverage": float(draws["coverage"].mean()),
                "mc_se": mc_standard_error(draws["coverage"]),
                "mean_ess": float(draws["ess"].mean()),
                "mean_max_normalised_weight": float(draws["max_weight"].mean()),
                "mean_inf_fraction": float(draws["inf_fraction"].mean()),
            }
        w, p = res["weighted"], res["split"]
        out.append(
            {
                "setup": "source x~N(0,1), target x~N(1,1), y=x+exp(x/2)eps, true weights",
                "n_cal": SHIFT_CASE["n"],
                "n_test_per_resample": SHIFT_CASE["n_test"],
                "alpha": alpha,
                "n_resamples": N_RESAMPLES,
                "arms": res,
                "verdict": (
                    "VALID"
                    if w["mean_coverage"] >= 1 - alpha - 3 * w["mc_se"]
                    and p["mean_coverage"] < 1 - alpha - 10 * p["mc_se"]
                    else "DEVIATES"
                ),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the study, write the JSON, return 0 iff every case is VALID."""
    args = parse_args(argv)
    split_cases = _split_cases()
    failure_cases = _failure_cases()
    shift_cases = _shift_cases()
    all_cases = [*split_cases, *failure_cases, *shift_cases]
    n_valid = sum(c["verdict"] == "VALID" for c in all_cases)
    result = {
        "schema": RESULT_SCHEMA,
        "kind": "validity",
        "subject": "poseconf.conformal (split, weighted) on synthetic scores with known laws",
        "data": "synthetic only; no SPEED+, no HIL, no P1",
        "oracle": False,
        "design": {
            "n_resamples": N_RESAMPLES,
            "band_level": BAND_LEVEL,
            "ks_min_pvalue": KS_MIN_P,
            "coverage": "exact conditional coverage from the score CDF, one draw per resample",
            "reference_law": "Beta(n+1-l, l), l = floor((n+1) alpha)",
        },
        "n_cases": len(all_cases),
        "n_valid": n_valid,
        "split_beta_law": split_cases,
        "failure_conventions": failure_cases,
        "weighted_covariate_shift": shift_cases,
        "provenance": {
            "poseconf_git_sha": poseconf_git_sha(),
            "created_at": utc_now_iso(),
            "seed": SEED,
            "numpy": np.__version__,
            "script": "scripts/run_conformal_synthetic_validity.py",
        },
    }
    write_result_json(args.out, result)
    print(f"{n_valid}/{len(all_cases)} cases VALID -> {args.out}")
    for c in all_cases:
        if c["verdict"] != "VALID":
            print(f"DEVIATES: {c}", file=sys.stderr)
    return 0 if n_valid == len(all_cases) else 1


if __name__ == "__main__":
    sys.exit(main())
