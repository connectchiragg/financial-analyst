"""Source-grounded financial analysis with deterministic calculations."""

from .analytics import ComparisonError, RevenueComparison, RevenueObservation, compare_revenue

__all__ = ["ComparisonError", "RevenueComparison", "RevenueObservation", "compare_revenue"]
