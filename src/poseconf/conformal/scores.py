"""Nonconformity scores A1-A3, B1-B2, C1-C2 (IMPLEMENTATION_PLAN.md section 1.3).

Each score is a pure `(predictions, labels) -> s` plus an inverse `(predictions, q) -> set` that never
takes labels. Implemented in Phase 1.
"""
