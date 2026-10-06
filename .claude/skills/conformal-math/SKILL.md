---
name: conformal-math
description: Reference formulas and pitfalls for split, weighted, Mondrian and adaptive conformal prediction, coverage evaluation, SO(3) sets, and beta-NLL. Use before writing, reviewing or testing any calibration, score, coverage or variance-head code in this repo.
user-invocable: true
---

# Conformal reference for `poseconf`

Notation: calibration scores s_1..s_n, miscoverage level alpha, test score s_test.

## Split conformal (the default arm)
- Index k = ceil((n + 1) * (1 - alpha)). If k > n, the quantile is +inf. Otherwise q = the k-th
  smallest calibration score (1-indexed, i.e. sorted[k - 1]).
- Set: C(x) = { y : s(x, y) <= q }. Coverage P(s_test <= q) >= 1 - alpha, and if scores are
  continuous with no ties, < 1 - alpha + 1/(n + 1).
- NOT the same as `np.quantile(s, 1 - alpha)`. With `method="higher"` at level k / n it is
  equivalent, but the k > n case still has to return +inf explicitly.
- Infinite scores: +inf (answer-required failure) and -inf (abstain-allowed failure) sort normally;
  do not filter them out before computing k.

## Coverage given the calibration set (validity test)
- Conditional on the calibration set, coverage ~ Beta(n + 1 - l, l), l = floor((n + 1) * alpha).
  Mean = (n + 1 - l)/(n + 1). Use this as the reference distribution for repeated re-splits
  (KS test) — Angelopoulos & Bates §3.2.
- Monte-Carlo standard error of a mean coverage over R re-splits: sd(coverages) / sqrt(R).
- Clopper–Pearson 95 % CI for k covered out of m: [Beta.ppf(0.025, k, m - k + 1),
  Beta.ppf(0.975, k + 1, m - k)], with 0 / 1 at the edges.

## Weighted conformal (covariate shift, Tibshirani et al. 2019)
- Weights w(x) = dP_target / dP_source (x). From a probabilistic classifier c(x) = P(target | x)
  trained with n_s source and n_t target points: w(x) = c / (1 - c) * (n_s / n_t).
- For a test point x: p_i = w(x_i) / (sum_j w(x_j) + w(x)), p_test = w(x) / (same sum).
  q(x) = the smallest s such that sum_{i: s_i <= s} p_i >= 1 - alpha, with the p_test mass placed
  at +inf. The quantile is per test point.
- Effective sample size ESS = (sum w)^2 / sum w^2. As the classifier AUC -> 1, ESS -> small and
  q -> +inf. That is the method working, not failing.
- Clipping weights trades validity for size; if used, choose the clip on val_tune and say so.

## Mondrian / group-conditional
- Run split CP separately within each group g, with k_g from n_g. Gives coverage >= 1 - alpha
  within each group. Groups must be defined by test-time-observable quantities. A group with
  n_g < ceil(1/alpha) - 1 has q_g = +inf.

## Adaptive conformal inference (Gibbs & Candès 2021) — stretch
- alpha_{t+1} = alpha_t + gamma * (alpha - err_t), err_t = 1 if y_t not in C_t. Long-run average
  miscoverage -> alpha for any sequence. Requires the label after each step; SPEED+ HIL frames are
  not a trajectory, so the order is synthetic and must be said.

## Barber et al. 2023 coverage-gap bound (to explain HIL misses)
- Coverage >= 1 - alpha - (sum_i w_i * d_TV(Z, Z^i)) / (1 + sum_i w_i); large shift -> large
  possible gap. Cite as explanation, not as a computed number unless you compute it.

## SO(3) and translation sets
- Geodesic angle d(R1, R2) = 2 * arccos(clip(|<q1, q2>|, 0, 1)) in radians.
- Rotation ball {R : d(R, R_hat) <= r}; membership test with the same formula.
- Translation ball normalised by the PREDICTED range: { t : ||t - t_hat|| <= q * c_t * ||t_hat|| }.
- Joint pose score max(a / c_a, b / c_b) gives the product set with radii q * c_a, q * c_b and
  simultaneous coverage. Normalisers c_* are medians on val_tune solved frames.
- Keypoint joint score max_k s_k gives simultaneous coverage of all K keypoint sets.

## Mahalanobis keypoint score
- s_k = sqrt(r_k^T Sigma_k^{-1} r_k). Set = ellipse { y : (y - y_hat)^T Sigma^{-1} (y - y_hat) <= q^2 }.
- For an exactly calibrated 2-D Gaussian, P(s <= 1) = 1 - exp(-1/2) ≈ 0.393, P(s <= 2) ≈ 0.865.
- Crop -> full frame: Sigma_full = A^{-1} Sigma_crop A^{-T}, A = 2x2 linear part of the
  full-frame -> crop affine.

## beta-NLL (Seitzer et al. 2022)
- Per sample: L = stopgrad(sigma^(2 beta)) * NLL_gaussian(r; Sigma). beta = 0.5 default; beta = 0
  is plain NLL, beta = 1 behaves like MSE for the mean. Here the mean is frozen, so beta only
  re-weights which residuals the variance fits.

## Pitfalls checklist
- Calibrating and evaluating on overlapping frames.
- Fitting a normaliser / bin edge / clip on val_cal.
- Dropping PnP failures.
- Using GT range, GT box or pixel_error_mean inside a score.
- Reporting coverage without n, CI, answer rate, convention.
- Reading near-nominal coverage as success when the set is +inf.
