# Progress

| Phase | Status | Key numbers | Tag |
|---|---|---|---|
| 0 Scaffold | **complete** 2026-10-06 | 4/4 P1 checkpoint SHA-256 match the release manifest ([`results/p1_checkpoints.json`](../results/p1_checkpoints.json)); P1 pinned at `0f81b426` (same file, `provenance.p1_commit`) | `phase-0-complete` |
| 1 Conformal core ⚠️ | not started | | |
| 2 Splits + Level A | not started | | |
| 3 Dump + P1 parity | not started | | |
| 4 Level B | not started | | |
| 5 Variance head | not started | | |
| 6 Coverage under shift ⭐ | not started | | |
| 7 Head, ONNX, latency | not started | | |
| 8 Validation sweep | not started | | |
| 9 Publication | not started | | |

Update at the end of each phase via `/phase-gate N`.

## Phase 0 — Scaffold (complete, 2026-10-06)

- Package skeleton, `p1_adapter` (sole `speedpose` import), `provenance`, checkpoint verification,
  CI with `submodules: recursive`. Gate: `make lint && make test && make smoke` green locally and in
  CI on `976894b`; `import speedpose, poseconf` works.
- Checkpoints: `detector`, `keypoint_a1`, `keypoint_a2`, `direct_regression_a1` all match the
  `phase-9-complete` `SHA256SUMS.txt` — [`results/p1_checkpoints.json`](../results/p1_checkpoints.json).
- Split sizes confirmed against P1's audit (`external/spacecraft-pose-baseline/results/dataset_audit.json`,
  `counts`): synthetic train 47,966 / validation 11,994; lightbox 6,740; sunlamp 2,791. The
  `keypoint_a2` synthetic sidecar covers all 11,994 validation frames (`tests/test_adapter.py`).
- Carried into Phase 1: TorchCP is not installable alongside P1's numpy pin; MAPIE 1.5.0 is the
  remaining library oracle (`docs/DECISIONS.md`). Sidecar box columns exist only for
  `predicted_crop` (IoU) / `pipeline_gt_crop` (boxes), so the IoU slice is `predicted_crop`-only.
