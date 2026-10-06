# Limitations

Filled in Phase 8 from `VALIDATION_CHECKLIST.md` (verdict + evidence per line). Known before
starting:

- **Calibration data was used for P1 checkpoint selection.** Synthetic validation chose P1's
  `best.pt`, so it is not perfectly exchangeable with fresh synthetic data. Evidence for or against
  a practical effect: the Phase 2 Beta-law test.
- **Single pose model.** All conclusions are about P1 `keypoint_a2` (with `keypoint_a1` as a check),
  not about keypoint pipelines in general.
- **HIL is not flight.** `lightbox`/`sunlamp` are laboratory hardware-in-the-loop imagery.
- **Sunlamp is small after PnP failures.** P1 solves 13.0 % of sunlamp; poolB will have on the order
  of 180 answered frames.
- **Inherited ONNX parity miss.** P1's keypoint-stage parity gate is unmet; anything built on that
  export inherits it.
