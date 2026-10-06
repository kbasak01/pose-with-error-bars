# Validation checklist — `pose-with-error-bars`

Phase 8 walks every line. Each gets a verdict (✅ met · ⚠️ met with caveat · ❌ unmet) and a pointer to
evidence, recorded in `docs/LIMITATIONS.md`. An unmet line is reported, never deleted.

## A. Conformal correctness
- A1. `conformal_quantile` uses the `⌈(n+1)(1−α)⌉`-th order statistic; matches brute force for n ∈ {1,2,9,10,99,4798}.
- A2. Returns `+∞` when `⌈(n+1)(1−α)⌉ > n`; never clips to the max score.
- A3. NaN scores raise; `±∞` scores are handled per convention.
- A4. Quantiles agree with MAPIE (v1, prefit) and TorchCP on identical residuals.
- A5. Synthetic-data property tests: coverage over ≥2,000 resamples inside the Beta law's 99 % band, for Gaussian, heavy-tailed and heteroscedastic noise.
- A6. Weighted CP: test-point mass placed at `+∞`; coverage restored under a synthetic covariate shift with known weights; ESS reported.
- A7. Mondrian groups with too few points return `∞`, never borrow.
- A8. Every score's set function takes no label argument (signature test).
- A9. `src/poseconf/conformal/` imports no torch (test).

## B. Splits and leakage
- B1. Split manifests committed with SHA-256; regeneration byte-identical.
- B2. `val_tune`, `val_cal`, `val_test` pairwise disjoint and cover synthetic validation exactly once.
- B3. HIL poolA/poolB disjoint, each domain covered exactly once.
- B4. No code path reads a HIL label outside poolB evaluation or an `oracle_*` arm (grep + runtime assertion).
- B5. Only `weighted_unlabeled_target` reads poolA images; it never reads their labels.
- B6. Normalisers, Mondrian edges, `g(conf̄)`, weight clipping, variance-head selection all fit on `val_tune` only.
- B7. Nothing was chosen by looking at HIL coverage (DECISIONS.md and git history checked).
- B8. The P1 checkpoint-selection caveat (selected on all of synthetic validation) is disclosed with the Phase 2 Beta-law evidence.

## C. Reproduction of P1
- C1. Submodule pinned at `phase-9-complete`; no local modifications (`git -C external/... status` clean).
- C2. Checkpoint SHA-256 match P1's release manifest.
- C3. Dump success flags agree with P1 sidecars ≥ 99.9 % per domain; median |Δe_r| ≤ 1e-6 rad.
- C4. Dump solve rates match P1's committed JSON.

## D. Variance head
- D1. P1 trunk parameters hash-identical after training.
- D2. Coordinates bit-identical with/without the head.
- D3. Covariances positive-definite for every output; σ clamps recorded.
- D4. Overfit-one-batch gate passed.
- D5. Selection on `val_tune` only; reported metrics on `val_test`.
- D6. Two seeds; spread reported.
- D7. NLL and set size vs heatmap-moment baseline reported whichever wins.

## E. Evaluation rigour
- E1. Every table states convention (answer-required / abstain-allowed), arm tag, α, n.
- E2. Coverage has a Clopper–Pearson 95 % interval everywhere.
- E3. Answer rate and silent-failure rate beside every coverage number.
- E4. `∞` set sizes reported, not dropped.
- E5. Oracle rows (`gt_crop`, `oracle_*`) in separate, labelled table sections.
- E6. In-distribution repeated-split coverage matches the Beta law (KS) or the deviation is explained.
- E7. Conditional coverage slices reported for every domain.
- E8. Sunlamp numbers always carry n (answered frames) and CI.
- E9. Same P1 PnP configuration in every row (read from the pinned config, recorded in each JSON).

## F. Propagation and geometry
- F1. PURSE membership ≡ joint keypoint coverage (numerical test).
- F2. Linearised Jacobian checked against finite differences.
- F3. Sampled radius labelled as an inner approximation (lower bound).
- F4. `pose-geometry-verifier` sign-off on `propagate.py`.

## G. Export and deployment
- G1. ONNX opset 17, `.eval()` before export, named I/O, checker passes, no `ArgMax/TopK/NonZero`.
- G2. Variance outputs tensor parity < 1e-4 fp32 (TF32 off); fp16 relaxation stated with measured delta.
- G3. Inherited P1 keypoint parity miss stated, not hidden.
- G4. Latency with ≥50 warmup / ≥500 timed, p50/p90/p99/max, provider verified per row.
- G5. Calibration JSON round-trips; loader refuses a mismatched checkpoint hash.
- G6. A4000-is-not-a-Jetson caveat on every latency table.

## H. Claims and honesty
- H1. Every README number traces to a committed `results/` file (claim-trace table).
- H2. No "guaranteed / certified / provably / safe" applied to HIL results.
- H3. "What the guarantee says and does not say" box present in README.
- H4. Prior art (Yang & Pavone 2023, Wang et al. 2025) cited and the contribution stated relative to it.
- H5. Limitations section lists single-model scope, simulation/HIL-only, exchangeability caveat, small sunlamp n.

## I. Repository quality
- I1. `make smoke` green on a clean clone without the dataset or a GPU (CI proves it).
- I2. Licences: MIT code; SPEED+ CC BY-NC-SA 4.0 attributed, not redistributed; HIL imagery not in README figures unless NC-SA terms are stated on the figure.
- I3. No dumps, checkpoints, ONNX or large binaries in git.
- I4. Release assets with `SHA256SUMS.txt`; hashes also in `results/`.

## J. Ship-deck reusability
- J1. `ConformalPoseHead` works from scores + predictions only; no SPEED+-specific constants in `conformal/`.
- J2. A worked example in `docs/` shows recalibrating on a new target with a different wireframe (synthetic data).
