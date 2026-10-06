"""Split conformal quantile: the ceil((n+1)(1-alpha))-th order statistic, +inf past n.

Handles +/-inf scores (PnP-failure conventions) and refuses NaN loudly. Implemented in Phase 1.
"""
