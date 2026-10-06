# Required sources

Checked 2026-10-06. Where a version or code release is stated, re-check it at install time.

## Upstream code and data (already on the workstation)

| Source | What P6 takes from it | Licence |
|---|---|---|
| `github.com/kbasak01/spacecraft-pose-baseline` @ `phase-9-complete` (`0f81b426e0a7a7ec26369fb31ad1ea19161434a1`) | `speedpose` package, configs, `pose_convention.yaml`, wireframe, committed per-sample sidecars `results/<run>/<domain>[_<arm>]_samples.npz` | MIT (code) |
| P1 release `phase-9-complete` | `best.pt` for `detector`, `keypoint_a1`, `keypoint_a2`, `direct_regression_a1`; `SHA256SUMS.txt` | weights carry SPEED+ NC/SA terms |
| **SPEED+**, Stanford Digital Repository, `purl.stanford.edu/wv398fc4383` | images + labels; synthetic 47,966 train / 11,994 val, `lightbox` 6,740, `sunlamp` 2,791 | **CC BY-NC-SA 4.0** |

Note: the project-ideas document lists SPEED+ as "CC BY 4.0". That is wrong — P1 verified the
dataset's own `LICENSE.md` reads Attribution-NonCommercial-ShareAlike 4.0 (P1 `docs/DECISIONS.md`).
Use NC-SA everywhere.

The same document cites the conformal pose reference as "Yang & Carlone, CVPR 2023". The CVPR 2023
paper (arXiv:2303.12246) is by **Heng Yang and Marco Pavone**; cite it as Yang & Pavone.

Per-sample sidecar fields available without a GPU: `filename, success, is_outlier, n_inliers,
failure_reason, range_m, reprojection_rmse, e_t, e_t_normalised, e_r, e_pose, q_pred, t_pred,
confidence_mean, pixel_error_mean, bbox_used, bbox_gt, bbox_iou`. Of these, `range_m`,
`pixel_error_mean`, `bbox_gt`, `bbox_iou` use ground truth and may only slice evaluations.

## Papers — read in this order

1. **Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free
   Uncertainty Quantification"**, arXiv:2107.07511. Split CP, the finite-sample quantile, the Beta
   law of conditional coverage used as Phase 2's validity test, conditional-coverage diagnostics.
2. **Yang & Pavone, "Object Pose Estimation with Statistical Guarantees: Conformal Keypoint
   Detection and Geometric Uncertainty Propagation"**, CVPR 2023, arXiv:2303.12246. Conformal
   keypoint sets → pose uncertainty set (PURSE), RANSAG average pose, SDP worst-case bounds. The
   reference for Level B.
3. **Wang, Li, Wang, Guan, Shang & Yu, "Deterministic Object Pose Confidence Region Estimation"**,
   ICCV 2025, arXiv:2506.22720. Conformal keypoint regions + implicit-function-theorem propagation
   to ellipsoidal pose regions; evaluated on LMO and the original **SPEED**. Closest prior art —
   the reference for the linearised estimator in Phase 4 and the comparison point for set volume.
   Code was announced as forthcoming at publication; check before relying on it.
4. **Gao, Tang, Qi & Yang, "CLOSURE: Fast Quantification of Pose Uncertainty Sets"**, RSS 2024,
   arXiv:2403.09990.
   Minimum enclosing geodesic ball of a PURSE — context for why the sampled radius in Phase 4 is an
   inner approximation.
5. **Tibshirani, Barber, Candès & Ramdas, "Conformal Prediction Under Covariate Shift"**,
   NeurIPS 2019, arXiv:1904.06019. Weighted CP via likelihood ratio — the
   `weighted_unlabeled_target` arm.
6. **Barber, Candès, Ramdas & Tibshirani, "Conformal prediction beyond exchangeability"**,
   Annals of Statistics 2023, arXiv:2202.13415. The coverage-gap bound in terms of total-variation
   distance — use it to *explain* the HIL miss.
7. **Gibbs & Candès, "Adaptive Conformal Inference Under Distribution Shift"**, NeurIPS 2021,
   arXiv:2106.00170. Stretch arm `aci_online`.
8. **Romano, Patterson & Candès, "Conformalized Quantile Regression"**, NeurIPS 2019,
   arXiv:1905.03222. Stretch CQR on the direct head.
9. **Seitzer, Tavakoli, Antic & Martius, "On the Pitfalls of Heteroscedastic Uncertainty
   Estimation with Probabilistic Neural Networks"**, ICLR 2022, arXiv:2203.09168. β-NLL for the
   variance head.
10. **Angelopoulos, Bates, Fisch, Lei & Schuster, "Conformal Risk Control"**, arXiv:2208.02814.
    Stretch: control expected capped `E_pose` instead of coverage.
11. **Van der Linden, Timans & Bekkers, "CP²: Leveraging Geometry for Conformal Prediction via
    Canonicalization"**, UAI 2025 (PMLR v286), code `github.com/computri/geometric_cp`.
    Related work: geometric shift and conformal validity.
12. SPEED+ papers already cited by P1: Park et al. 2022 (SPEED+, DOI 10.1109/AERO53065.2022.9843439),
    Park & D'Amico 2022 (SPNv2, arXiv:2203.04275) for the HIL metric definitions.

## Libraries

| Package | Role | Note |
|---|---|---|
| numpy, scipy | the entire conformal core | `scipy.stats.beta` for Beta law and Clopper–Pearson |
| scikit-learn | logistic domain classifier, isotonic `g(conf̄)` | |
| torch, timm, onnx, onnxruntime-gpu, tensorrt | inherited from P1's pinned stack | do not re-pin |
| MAPIE ≥ 1.0 (1.5.0 on PyPI, 2026-10-06) | **test oracle** for split-CP quantiles | v1 API: `SplitConformalRegressor`, `ConformalizedQuantileRegressor`; supports prefit estimators |
| TorchCP ≥ 1.2 (1.2.1, 2025-10-14) | **test oracle**; reference ACI implementation | |
| opencv | `projectPoints` Jacobian, iterative PnP for sampling | via P1's pinned wheel |

Neither MAPIE nor TorchCP is imported from `src/`: the core is small enough to own, must be
framework-free for the ship-deck port, and the oracles are more useful as independent checks.
