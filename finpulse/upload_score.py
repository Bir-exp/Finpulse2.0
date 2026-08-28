"""Transparent behavioral score for reviewed uploaded statements.

This score is intentionally separate from the FinPulse v1 synthetic-user
score.  It uses only the declared Monthly Available Amount and behavioral
features derived from included debit transactions.  Statement credits are
never score inputs.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Mapping

import math

from scripts.score_engine import score_band


def _ratio(features: Mapping[str, Any], name: str) -> float:
    value = features.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric ratio")
    ratio = float(value)
    if not math.isfinite(ratio):
        raise ValueError(f"{name} must be a finite numeric ratio")
    return ratio


def _tier(
    value: float,
    tiers: tuple[tuple[float, int, str], ...],
    fallback: tuple[int, str],
) -> tuple[int, str]:
    for upper_bound, points, label in tiers:
        if value <= upper_bound:
            return points, label
    return fallback


def _investment_points(ratio: float) -> tuple[int, str]:
    if ratio >= 0.20:
        return 20, "At least 20% allocated explicitly"
    if ratio >= 0.15:
        return 17, "15% to under 20% allocated explicitly"
    if ratio >= 0.10:
        return 14, "10% to under 15% allocated explicitly"
    if ratio >= 0.05:
        return 9, "5% to under 10% allocated explicitly"
    if ratio > 0:
        return 4, "Some explicit saving or investment allocation"
    return 0, "No explicit saving or investment transaction observed"


def _remainder_points(ratio: float) -> tuple[int, str]:
    if ratio >= 0.30:
        return 15, "At least 30% estimated unspent capacity"
    if ratio >= 0.20:
        return 12, "20% to under 30% estimated unspent capacity"
    if ratio >= 0.10:
        return 9, "10% to under 20% estimated unspent capacity"
    if ratio >= 0:
        return 5, "Non-negative estimated unspent capacity below 10%"
    return 0, "Included debit spending exceeds the declared capacity"


def _budget_utilization(budget_summary: Mapping[str, Any] | None) -> float | None:
    if budget_summary is None:
        return None
    value = budget_summary.get("budget_utilization")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("budget_utilization must be a finite numeric ratio")
    utilization = float(value)
    if not math.isfinite(utilization) or utilization < 0:
        raise ValueError("budget_utilization must be a non-negative finite ratio")
    return utilization


def calculate_upload_behavioral_score(
    features: Mapping[str, Any],
    budget_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the dedicated 40/35/25 uploaded-statement score.

    Optional budget information refines the eight-point discipline
    subcomponent by applying the stricter of observed capacity overspending and
    budget adherence.  When no budget is supplied, all eight points remain
    available from observable overspending-month behavior.
    """

    expense_ratio = _ratio(features, "avg_expense_ratio")
    desire_ratio = _ratio(features, "avg_desire_ratio")
    overspending_ratio = _ratio(features, "overspending_month_ratio")
    investment_ratio = _ratio(features, "avg_investment_ratio")
    remaining_ratio = _ratio(features, "avg_remaining_ratio")
    repayment_ratio = _ratio(features, "avg_repayment_ratio")

    overall_points, overall_label = _tier(
        expense_ratio,
        (
            (0.60, 20, "Spending is at or below 60% of capacity"),
            (0.75, 18, "Spending is above 60% and at or below 75% of capacity"),
            (0.90, 14, "Spending is above 75% and at or below 90% of capacity"),
            (1.00, 9, "Spending is above 90% and at or below capacity"),
            (1.10, 4, "Spending is up to 10% above capacity"),
        ),
        (0, "Spending is more than 10% above capacity"),
    )
    desire_points, desire_label = _tier(
        desire_ratio,
        (
            (0.05, 12, "Desire spending is at or below 5% of capacity"),
            (0.10, 10, "Desire spending is above 5% and at or below 10%"),
            (0.15, 7, "Desire spending is above 10% and at or below 15%"),
            (0.25, 3, "Desire spending is above 15% and at or below 25%"),
        ),
        (0, "Desire spending is above 25% of capacity"),
    )
    overspending_points, overspending_label = _tier(
        overspending_ratio,
        (
            (0.00, 8, "No observed month exceeds declared capacity"),
            (0.15, 6, "Up to 15% of observed months exceed capacity"),
            (0.30, 4, "More than 15% and up to 30% of months exceed capacity"),
            (0.50, 2, "More than 30% and up to 50% of months exceed capacity"),
        ),
        (0, "More than half of observed months exceed capacity"),
    )

    budget_utilization = _budget_utilization(budget_summary)
    budget_points: int | None = None
    budget_label: str | None = None
    discipline_points = overspending_points
    discipline_basis = "Observed capacity overspending; no budget was supplied"
    if budget_utilization is not None:
        budget_points, budget_label = _tier(
            budget_utilization,
            (
                (0.90, 8, "Spending uses at most 90% of the supplied budget"),
                (1.00, 6, "Spending is within the supplied budget"),
                (1.10, 3, "Spending is up to 10% above the supplied budget"),
            ),
            (0, "Spending is more than 10% above the supplied budget"),
        )
        discipline_points = min(overspending_points, budget_points)
        discipline_basis = (
            "Stricter of observed capacity overspending and supplied-budget adherence"
        )

    spending_score = overall_points + desire_points + discipline_points
    investment_points, investment_label = _investment_points(investment_ratio)
    remainder_points, remainder_label = _remainder_points(remaining_ratio)
    saving_score = investment_points + remainder_points
    repayment_points, repayment_label = _tier(
        repayment_ratio,
        (
            (0.05, 25, "Repayments are at or below 5% of capacity"),
            (0.10, 23, "Repayments are above 5% and at or below 10%"),
            (0.20, 19, "Repayments are above 10% and at or below 20%"),
            (0.30, 13, "Repayments are above 20% and at or below 30%"),
            (0.40, 7, "Repayments are above 30% and at or below 40%"),
        ),
        (3, "Repayments are above 40% of capacity"),
    )

    total_score = max(0, min(100, spending_score + saving_score + repayment_points))
    return {
        "total_score": int(total_score),
        # Retained as a presentation-compatible alias while callers migrate to
        # the explicit upload-score name above.
        "finpulse_score": int(total_score),
        "score_band": score_band(total_score),
        "components": {
            "spending_control": {
                "score": spending_score,
                "max_score": 40,
                "explanation": (
                    "Measures total included-debit spending, Desire spending, "
                    "and observed spending discipline against declared capacity."
                ),
                "subcomponents": {
                    "overall_spending": {
                        "score": overall_points,
                        "max_score": 20,
                        "ratio": expense_ratio,
                        "explanation": overall_label,
                    },
                    "desire_control": {
                        "score": desire_points,
                        "max_score": 12,
                        "ratio": desire_ratio,
                        "explanation": desire_label,
                    },
                    "budget_and_overspending_discipline": {
                        "score": discipline_points,
                        "max_score": 8,
                        "overspending_month_ratio": overspending_ratio,
                        "overspending_score": overspending_points,
                        "overspending_explanation": overspending_label,
                        "budget_utilization": budget_utilization,
                        "budget_score": budget_points,
                        "budget_explanation": budget_label,
                        "explanation": discipline_basis,
                    },
                },
            },
            "saving_investment": {
                "score": saving_score,
                "max_score": 35,
                "explanation": (
                    "Combines explicit Investment/Savings debits with estimated "
                    "unspent capacity; the remainder is not claimed as confirmed savings."
                ),
                "subcomponents": {
                    "explicit_investment_allocation": {
                        "score": investment_points,
                        "max_score": 20,
                        "ratio": investment_ratio,
                        "explanation": investment_label,
                    },
                    "estimated_unspent_capacity": {
                        "score": remainder_points,
                        "max_score": 15,
                        "ratio": remaining_ratio,
                        "explanation": remainder_label,
                    },
                },
            },
            "repayment_management": {
                "score": repayment_points,
                "max_score": 25,
                "explanation": (
                    "Measures repayment burden relative to Monthly Available Amount; "
                    "it does not infer creditworthiness or default risk."
                ),
                "subcomponents": {
                    "repayment_burden": {
                        "score": repayment_points,
                        "max_score": 25,
                        "ratio": repayment_ratio,
                        "explanation": repayment_label,
                    }
                },
            },
        },
        "method": (
            "Dedicated FinPulse 2.0 uploaded-statement behavioral score: "
            "Spending Control 40, Saving & Investment Behavior 35, and "
            "Repayment Management 25. Credits and Stability are not score inputs."
        ),
    }
