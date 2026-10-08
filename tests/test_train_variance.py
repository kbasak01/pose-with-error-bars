"""β-NLL loss, the overfit-gate floor, and the head-quality metrics (Phase 5)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.stats import multivariate_normal

from poseconf import p1_adapter
from poseconf.engine import head_metrics
from poseconf.engine.train_variance import (
    beta_nll,
    chol_stats,
    gaussian_nll,
    homoscedastic_floor,
)
from poseconf.models.variance_head import HeadConfig, chol_to_cov

HEAD = HeadConfig(256, 16, 11, 0.05, 64.0, 0.99, 2.0)


def _random_chol(n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    sx, sy = rng.uniform(0.3, 5.0, n), rng.uniform(0.3, 5.0, n)
    rho = rng.uniform(-0.9, 0.9, n)
    return torch.tensor(np.stack([sx, rho * sy, sy * np.sqrt(1 - rho**2)], -1))


def test_gaussian_nll_matches_scipy_and_numpy():
    chol = _random_chol(50, 0)
    residual = torch.tensor(np.random.default_rng(1).normal(size=(50, 2)) * 3)
    cov = chol_to_cov(chol).numpy()
    want = np.array(
        [
            -multivariate_normal(np.zeros(2), c).logpdf(r)
            for c, r in zip(cov, residual.numpy(), strict=True)
        ]
    )
    np.testing.assert_allclose(gaussian_nll(chol, residual).numpy(), want, rtol=1e-10)
    np.testing.assert_allclose(head_metrics.gaussian_nll(cov, residual.numpy()), want, rtol=1e-10)


def test_beta_zero_is_plain_nll_and_beta_weight_is_detached():
    chol = _random_chol(20, 2).requires_grad_(True)
    residual = torch.tensor(np.random.default_rng(3).normal(size=(20, 2)))
    assert torch.equal(beta_nll(chol, residual, 0.0), gaussian_nll(chol, residual))

    beta = 0.5
    beta_nll(chol, residual, beta).sum().backward()
    got = chol.grad.clone()
    chol.grad = None
    weight = (chol[:, 0] * chol[:, 2]).detach() ** beta
    (weight * gaussian_nll(chol, residual)).sum().backward()
    torch.testing.assert_close(got, chol.grad)
    # weight = det(Σ)^{β/2}
    det = torch.linalg.det(chol_to_cov(chol.detach()))
    torch.testing.assert_close(weight, det ** (beta / 2))


def test_masked_keypoints_do_not_change_the_loss():
    chol = _random_chol(12, 4).reshape(3, 4, 3)
    residual = torch.tensor(np.random.default_rng(5).normal(size=(3, 4, 2)))
    mask = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    loss = p1_adapter.masked_mean(beta_nll(chol, residual, 0.5), mask)
    residual2 = residual.clone()
    residual2[~mask] = 1e3
    assert torch.equal(loss, p1_adapter.masked_mean(beta_nll(chol, residual2, 0.5), mask))
    expected = beta_nll(chol, residual, 0.5)[mask].mean()
    torch.testing.assert_close(loss, expected)


def test_homoscedastic_floor_is_the_mle_nll():
    rng = np.random.default_rng(6)
    residual = rng.multivariate_normal([0, 0], [[4.0, 1.0], [1.0, 2.0]], size=500)
    floor = homoscedastic_floor(residual)
    second = residual.T @ residual / len(residual)
    nll = head_metrics.gaussian_nll(np.broadcast_to(second, (500, 2, 2)), residual).mean()
    assert floor == pytest.approx(nll, rel=1e-12)
    for factor in (0.9, 1.1):
        worse = head_metrics.gaussian_nll(np.broadcast_to(second * factor, (500, 2, 2)), residual)
        assert worse.mean() > floor
    with pytest.raises(ValueError):
        homoscedastic_floor(np.zeros((10, 2)))


def test_global_scale_minimises_nll():
    rng = np.random.default_rng(7)
    cov = chol_to_cov(_random_chol(400, 8)).numpy()
    residual = np.stack([rng.multivariate_normal([0, 0], 3.0 * c) for c in cov])
    s = head_metrics.global_scale(cov, residual)
    best = head_metrics.gaussian_nll(cov * s, residual).mean()
    for factor in (0.95, 1.05):
        assert head_metrics.gaussian_nll(cov * s * factor, residual).mean() > best
    assert s == pytest.approx(3.0, rel=0.15)


def test_head_quality_on_exact_gaussians():
    rng = np.random.default_rng(9)
    n, k = 4000, 3
    cov = chol_to_cov(_random_chol(n * k, 10)).numpy().reshape(n, k, 2, 2)
    residual = np.stack([rng.multivariate_normal([0, 0], c) for c in cov.reshape(-1, 2, 2)])
    residual = residual.reshape(n, k, 2)
    mask = np.ones((n, k), dtype=bool)
    mask[0, 0] = False
    out = head_metrics.head_quality(cov, residual, mask, confidence=rng.normal(size=(n, k)))
    ref = head_metrics.GAUSSIAN_2D_COVERAGE
    assert out["coverage_1sigma"] == pytest.approx(ref["1sigma"], abs=0.015)
    assert out["coverage_2sigma"] == pytest.approx(ref["2sigma"], abs=0.01)
    assert out["global_scale"] == pytest.approx(1.0, abs=0.05)
    assert out["spearman_sigma_vs_error"] > 0.3  # σ varies, so it ranks the error
    assert len(out["spearman_sigma_vs_error_per_keypoint"]) == k
    assert out["n_keypoints"] == n * k - 1
    assert out["global_scale_fitted_on_this_split"]
    fixed = head_metrics.head_quality(cov, residual, mask, scale=2.0)
    assert not fixed["global_scale_fitted_on_this_split"] and fixed["global_scale"] == 2.0


def test_head_quality_refuses_non_pd():
    cov = np.zeros((2, 1, 2, 2))
    with pytest.raises(ValueError, match="positive definite"):
        head_metrics.head_quality(cov, np.zeros((2, 1, 2)), np.ones((2, 1), dtype=bool))


def test_chol_stats_counts_clamps_and_saturation():
    s_min, s_max = HEAD.sigma_min, HEAD.sigma_max
    rho = 0.99
    chol = np.array(
        [
            [[s_min, 0.0, s_min]],
            [[s_max, 0.0, s_max]],
            [[1.0, rho * 2.0, 2.0 * math.sqrt(1 - rho**2)]],
            [[1.0, 0.0, 1.0]],
        ]
    )
    out = chol_stats(chol, np.ones((4, 1), dtype=bool), HEAD, rho_saturation=0.98)
    assert out["n"] == 4
    assert out["frac_sigma_at_min"] == pytest.approx(2 / 8)
    assert out["frac_sigma_at_max"] == pytest.approx(2 / 8)
    assert out["frac_rho_saturated"] == pytest.approx(1 / 4)


@pytest.mark.dataset
@pytest.mark.slow
def test_selection_reads_val_tune_only(local_paths):
    import importlib
    import sys
    from pathlib import Path

    from poseconf.data.splits import load_split

    if local_paths.crop_cache is None:
        pytest.skip("p1_crop_cache not set")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    script = importlib.import_module("train_variance_head")
    subset, counts = script.val_tune_dataset("keypoint_a2", paths=local_paths, seed=1337)
    names = p1_adapter.crop_dataset_filenames(subset.dataset)[subset.indices]
    assert set(names) <= set(load_split("synthetic_val_tune"))
    assert counts["served"] == len(names) and counts["manifest"] == 2399


def test_second_seed_config_differs_only_in_seed_and_run_name():
    """The Phase 8 seed spread compares like with like (VALIDATION_CHECKLIST D6)."""
    base = yaml.safe_load(Path("configs/variance_head.yaml").read_text(encoding="utf-8"))
    other = yaml.safe_load(Path("configs/variance_head_s2026.yaml").read_text(encoding="utf-8"))
    assert (base["seed"], other["seed"]) == (1337, 2026)
    assert other["run_name"] == "vhead_a2_s2026"
    for config in (base, other):
        del config["seed"], config["run_name"]
    assert base == other
