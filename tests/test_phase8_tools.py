"""Phase 8 helper scripts: the re-run comparison and the variance-head seed spread."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> ModuleType:
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(name)


# --------------------------------------------------------------------------------------------------
# check_rerun
# --------------------------------------------------------------------------------------------------


def test_json_comparison_ignores_provenance_only():
    rerun = _script("check_rerun")
    old = {"rows": [{"coverage": 0.9, "q": "inf"}], "provenance": {"poseconf_git_sha": "a-dirty"}}
    new = copy.deepcopy(old)
    new["provenance"] = {"poseconf_git_sha": "b", "created_at": "now"}
    n, diffs = rerun.json_differences(old, new)
    assert (n, diffs) == (2, [])
    new["rows"][0]["coverage"] = 0.9000000000000001
    assert rerun.json_differences(old, new)[1] == ["/rows[0]/coverage"]
    new["rows"][0]["coverage"] = 0.9
    new["rows"].append({})
    assert rerun.json_differences(old, new)[1] == ["/rows (length 1 vs 2)"]


def test_json_comparison_is_type_strict():
    rerun = _script("check_rerun")
    assert rerun.json_differences({"n": 1}, {"n": 1.0})[1] == ["/n"]


def test_npz_comparison_is_exact_and_nan_aware(tmp_path):
    rerun = _script("check_rerun")
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, x=np.array([1.0, np.nan]), k=np.arange(3))
    np.savez(b, x=np.array([1.0, np.nan]), k=np.arange(3))
    assert rerun.npz_differences(a, b) == (5, [])
    np.savez(b, x=np.array([1.0, np.nan]), k=np.arange(3) + 1)
    assert rerun.npz_differences(a, b)[1] == ["k"]


def test_main_exits_nonzero_when_a_pair_differs(tmp_path):
    rerun = _script("check_rerun")
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"v": 1}), encoding="utf-8")
    b.write_text(json.dumps({"v": 2}), encoding="utf-8")
    out = tmp_path / "out.json"
    code = rerun.main(["--pair", str(a), str(b), "--what", "test", "--out", str(out)])
    record = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and record["all_identical"] is False
    assert record["pairs"][0]["differences"] == ["/v"]


# --------------------------------------------------------------------------------------------------
# seed_spread
# --------------------------------------------------------------------------------------------------


def _level_c(run: str, sha: str, shift: float) -> dict:
    rows, comparison = [], []
    for convention in ("abstain_allowed", "answer_required"):
        for score in ("C1", "C2"):
            rows.append({
                "score": score, "convention": convention, "alpha": 0.1,
                "coverage": 0.9 + shift, "coverage_ci95": [0.89, 0.91], "n_total": 100,
                "n_answered": 93, "answer_rate": 0.93, "silent_failure_rate": 0.05,
                "quantile": 3.0 + shift, "resplits": {"verdict": "VALID"},
            })  # fmt: skip
        comparison.append({
            "pair": "C1_vs_B2", "convention": convention, "alpha": 0.1,
            "kp_px_median": [20.0 + shift, 16.0], "coverage": [0.9, 0.9],
        })  # fmt: skip
        comparison.append({
            "pair": "C2_vs_A1", "convention": convention, "alpha": 0.1,
            "rot_deg_median": [1.4, 2.3], "trans_frac_median": [0.014, 0.011],
            "coverage": [0.9, 0.9], "answer_rate": [0.93, 0.93],
        })  # fmt: skip
    learned = {
        "nll_mean": 3.0 + shift, "nll_mean_rescaled": 3.0, "coverage_1sigma": 0.52,
        "coverage_2sigma": 0.88, "spearman_sigma_vs_error": 0.52,
    }  # fmt: skip
    return {
        "variance_head": {"run_name": run, "checkpoint_sha256": sha},
        "head_quality": {"val_test": {"learned": learned}},
        "verdict_counts": {"VALID": 26, "DEGENERATE": 6, "DEVIATES": 0},
        "rows": rows,
        "comparison": comparison,
    }


def _training(run: str, sha: str, seed: int) -> dict:
    return {
        "run_name": run,
        "seed": seed,
        "best_checkpoint_sha256": sha,
        "best_epoch": 15,
        "best_val_tune_nll": 2.98,
    }


def test_seed_spread_is_the_range_of_each_quoted_number():
    spread_mod = _script("seed_spread")
    seeds = [
        spread_mod.seed_values(_level_c("s1", "aa", 0.0), _training("s1", "aa", 1337)),
        spread_mod.seed_values(_level_c("s2", "bb", 0.01), _training("s2", "bb", 2026)),
    ]
    spread = spread_mod.spread(seeds)
    assert spread["val_test_nll_mean"] == pytest.approx(0.01)
    assert spread["C1_abstain_allowed"]["coverage"] == pytest.approx(0.01)
    assert spread["C1_abstain_allowed"]["kp_px_median"] == pytest.approx(0.01)
    assert spread["C2_answer_required"]["rot_deg_median"] == 0.0
    assert "coverage_ci95" not in spread["C1_abstain_allowed"]
    assert {"run_name", "seed", "best_epoch", "verdict_counts"}.isdisjoint(spread)


def test_seed_spread_refuses_a_level_c_from_another_checkpoint():
    spread_mod = _script("seed_spread")
    with pytest.raises(ValueError, match="not the training best.pt"):
        spread_mod.seed_values(_level_c("s1", "aa", 0.0), _training("s1", "zz", 1337))


def test_seed_is_recovered_from_the_hashed_config_for_older_records():
    spread_mod = _script("seed_spread")
    from poseconf.provenance import sha256_file

    sha = sha256_file(REPO_ROOT / "configs" / "variance_head.yaml")
    old = {"run_name": "vhead_a2_s1337", "provenance": {"config_sha256": sha}}
    assert spread_mod._seed(old) == 1337
    with pytest.raises(ValueError, match="no config matches"):
        spread_mod._seed({"run_name": "x", "provenance": {"config_sha256": "0" * 64}})


def test_rerun_record_shows_changed_inputs_and_refuses_a_dirty_rerun(tmp_path):
    rerun = _script("check_rerun")
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    prov = {"poseconf_git_sha": "abc-dirty", "config_sha256": "1", "dump_sha256": "d"}
    a.write_text(json.dumps({"v": 1, "provenance": prov}), encoding="utf-8")
    b.write_text(
        json.dumps(
            {"v": 1, "provenance": {**prov, "poseconf_git_sha": "def", "config_sha256": "2"}}
        ),
        encoding="utf-8",
    )
    pair = rerun.compare(a, b)
    assert pair["identical"] and pair["rerun_poseconf_git_sha"] == "def"
    assert pair["provenance_keys_differing"] == {"config_sha256": {"committed": "1", "rerun": "2"}}
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit, match="clean tree"):
        rerun.main(["--pair", str(b), str(a), "--what", "t", "--out", str(out)])
