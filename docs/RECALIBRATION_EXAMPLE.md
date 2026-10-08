# Worked example: recalibrating for a new target

How to take `poseconf.conformal` to a different target, here a ship deck, without any SPEED+
code. The runnable version is [`scripts/recalibration_example.py`](../scripts/recalibration_example.py),
and every number below is read from its committed output,
[`results/validity/recalibration_example.json`](../results/validity/recalibration_example.json).
`tests/test_recalibration_example.py` runs it in CI and pins that it imports only
`poseconf.conformal` and `poseconf.provenance`: no torch, no P1, no dataset.

**The data are invented.** The example demonstrates the mechanics on simulated, exchangeable
frames. It says nothing about how a real deck pipeline will behave.

## The new target

| | Tango (this project) | Example deck |
|---|---|---|
| Keypoints | 11 (P1 wireframe) | 6 deck markers, one raised 1.5 m |
| Camera | SPEED+ `camera.json`, distorted | 1280 × 720, f = 900 px, no distortion |
| Range | P1's synthetic range | 30–80 m |
| Keypoint model | P1 `keypoint_a2` + variance head | a stand-in whose σ̂ is 1.6× too small and whose errors are Student-t(4) |

Everything target-specific is an *input*: the wireframe and camera go into
`propagate.CameraGeometry`, and K is read from the arrays. Nothing in `conformal/` changes.

## The recipe

1. **Collect labelled calibration frames from the new target**, from the same distribution the
   deployment will see. For each frame, record: the predicted keypoints, the predicted
   covariances, which keypoints project inside the image (`include`), and whether PnP returned a
   pose (`valid`). Keep failures; never drop them.
2. **Fix the convention** before looking at any number. The example uses `abstain_allowed`: a
   failed frame scores −∞, and the deployment abstains on it.
3. **Score and take the quantile**, with labels used here and nowhere else:

   ```python
   scores = score_mahalanobis(y_hat, y_gt, include, valid, cov=cov, convention="abstain_allowed")
   q = conformal_quantile(scores, alpha)        # the ⌈(n+1)(1−α)⌉-th order statistic, or +inf
   ```

4. **Package it, locked to the models that produced the scores**:

   ```python
   artifact = build_artifact(score_id="C1", alpha=alpha, convention="abstain_allowed", quantile=q,
                             n_cal=len(scores), normalisers={}, quaternion_order="scalar_first",
                             p1_checkpoint_sha256=sha(keypoint_model),     # see "Lock slots"
                             detector_checkpoint_sha256=sha(roi_detector),
                             variance_head_sha256=sha(variance_head),
                             context={"crop_source": "predicted_crop", "target": "..."})
   ```

5. **Deploy**: load with the hashes of the checkpoints actually running, then call `predict` once
   per frame. `FrameEstimate` has no label field.

   ```python
   head = ConformalPoseHead.from_dict(artifact, p1_checkpoint_sha256=..., detector_sha256=...,
                                      variance_head_sha256=...)
   out = head.predict(FrameEstimate(valid, q_hat, t_hat, keypoints, conf, bbox), cov=cov)
   # Answer(set=KeypointSet ...)  or  Abstain(reason=...)
   ```

6. **Recalibrate whenever a locked model changes.** The loader refuses a mismatched checkpoint, and
   the example checks that: loading with a "retrained" keypoint model's hash raises
   `CalibrationMismatchError` (`refuses_retrained_model: true`).

## Lock slots (checklist J1)

The artifact's lock fields are named after this project's pipeline. A port fills them as follows,
and nothing else about them is P1-specific:

| Slot | Holds, in a port |
|---|---|
| `p1_checkpoint_sha256` | the keypoint model's checkpoint |
| `detector_checkpoint_sha256` | whatever chooses the crop (ROI detector), with `crop_source: predicted_crop` |
| `variance_head_sha256` | the covariance model (C1/C2 only) |

Do not use `crop_source: gt_crop` for a deployment: in this project it means the crop came from
the label, an oracle. Renaming the slots needs an artifact schema bump, which would re-hash every
committed calibration, so it was not done (`docs/LIMITATIONS.md`, J1).

## What it shows

`α = 0.10`, `abstain_allowed`, C1, `n_cal = 2,000`, test n = 5,000:

| | Coverage [95 % CI] | Answer rate | Silent failure |
|---|---|---|---|
| Conformal C1, marginal (abstentions count as covered) | 0.8910 [0.8820, 0.8995] | 0.9476 (4,738 answered) | 0.109 |
| Conformal C1, given answered | 0.8850 [0.8755, 0.8939] | | |
| Trusting σ̂ as a Gaussian (Bonferroni χ²₂), given answered | 0.3609 [0.3472, 0.3748] | | |

- **The miscalibrated σ̂ is the point.** Taken at face value it covers 36 % of answered frames
  where 90 % was meant. The conformal quantile, q = 5.387 against the Gaussian's 2.862, absorbs
  the miscalibration without retraining anything.
- **One draw sits below 0.90, and that is expected.** The guarantee is marginal over the
  calibration draw, and this table is one draw. Over 200 fresh calibration/test draws the mean
  coverage is 0.9000 (MC s.e. 0.0006). Draws range from 0.8734 to 0.9204, around a nominal 0.90
  and a continuous-score upper bound of 0.9005 (`repeated_draws`). The CI above covers test noise
  only, given this calibration set.
- **"Given answered" is a slice, not a guaranteed quantity.** Under `abstain_allowed` the guarantee
  is on the marginal row, with abstentions counted covered. Coverage given answered is only about
  1 − α / (answer rate) in expectation: 1 − 0.1 / 0.9476 ≈ 0.894 here. So 0.8850 is not a
  shortfall against 0.90. Under `answer_required` (finite q) an unanswered frame is uncovered. So
  marginal coverage ≥ 1 − α implies coverage given answered ≥ (1 − α) / (answer rate).
- **The set is a per-keypoint ellipse.** Its median largest radius is 21.3 px on answered frames.

## What it does not show

- Anything about a real deck. Exchangeability between calibration and deployment frames is the
  assumption the guarantee rests on, and this project's own headline result is how badly it fails
  under a synthetic → hardware shift (`results/TABLES.md`). Calibrate on frames from the
  conditions you will deploy in, and re-measure when they change.
- Pose-level sets. The example's frame records carry the true pose as the "estimate". C1 ignores
  it, but a pose-level score (A1–A3, C2) would need a real PnP estimate.
