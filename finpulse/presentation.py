"""Pure presentation helpers for the FinPulse Streamlit interface."""

from __future__ import annotations

from datetime import date
import math
from numbers import Real
from typing import Any, Mapping


SPENDING_CATEGORIES = (
    "Essentials",
    "Desire",
    "Repayment",
    "Investment/Savings",
    "Others",
)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _group_indian_digits(digits: str) -> str:
    if len(digits) <= 3:
        return digits
    tail = digits[-3:]
    head = digits[:-3]
    groups: list[str] = []
    while head:
        groups.append(head[-2:])
        head = head[:-2]
    return ",".join(reversed(groups)) + "," + tail


def format_inr(value: object) -> str:
    """Format a finite rupee amount using Indian digit grouping."""

    number = _finite_number(value)
    if number is None:
        return "Unavailable"
    rounded = round(abs(number), 2)
    whole, fraction = f"{rounded:.2f}".split(".")
    fraction = fraction.rstrip("0")
    amount = _group_indian_digits(whole)
    if fraction:
        amount += f".{fraction}"
    sign = "-" if number < 0 else ""
    return f"{sign}₹{amount}"


def format_percentage(value: object, *, already_percent: bool = False) -> str:
    """Format a ratio or percentage with one decimal place."""

    number = _finite_number(value)
    if number is None:
        return "Unavailable"
    percent = number if already_percent else number * 100
    return f"{percent:.1f}%"


def spending_metric_label(statement_summary: Mapping[str, Any]) -> str:
    """Use an explicitly averaged label when included debits span multiple months."""

    month_count = int(statement_summary.get("transaction_month_count") or 0)
    return "Average Monthly Spending" if month_count > 1 else "Monthly Spending"


def _parse_iso_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _period_name(start: date | None, end: date | None) -> str | None:
    if start is None or end is None:
        return None
    if start.year == end.year and start.month == end.month:
        return start.strftime("%b %Y")
    if start.year == end.year:
        return f"{start.strftime('%b')}–{end.strftime('%b %Y')}"
    return f"{start.strftime('%b %Y')}–{end.strftime('%b %Y')}"


def spending_period_context(statement_summary: Mapping[str, Any]) -> str:
    """Describe monthly display scope without exposing normalization mechanics."""

    month_count = int(statement_summary.get("transaction_month_count") or 0)
    period = _period_name(
        _parse_iso_date(statement_summary.get("start_date")),
        _parse_iso_date(statement_summary.get("end_date")),
    )
    if month_count > 1:
        base = f"Average per month based on {month_count} months of included debit transactions"
    else:
        base = "Based on the included debit transactions for this statement month"
    return f"{base} ({period})." if period else f"{base}."


def statement_period_context(cash_flow_summary: Mapping[str, Any]) -> str:
    """Describe the date scope of informational full-statement cash flow."""

    period = _period_name(
        _parse_iso_date(cash_flow_summary.get("statement_start_date")),
        _parse_iso_date(cash_flow_summary.get("statement_end_date")),
    )
    return (
        f"Full uploaded statement ({period}); informational only."
        if period
        else "Full uploaded statement; informational only."
    )


def build_overview_metrics(
    statement_summary: Mapping[str, Any],
    score_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Select existing analytics values for the non-technical Overview cards."""

    return {
        "monthly_available_amount": statement_summary.get("monthly_available_amount"),
        "monthly_spending": statement_summary.get(
            "monthly_normalized_debit_spending"
        ),
        "estimated_amount_left": statement_summary.get(
            "remaining_amount_estimate"
        ),
        "spending_label": spending_metric_label(statement_summary),
        "period_context": spending_period_context(statement_summary),
        "score": score_result.get("finpulse_score"),
        "score_band": score_result.get("score_band"),
        "provisional": bool(score_result.get("is_provisional", False)),
    }


def largest_spending_category(
    category_summary: Mapping[str, Mapping[str, Any]],
) -> tuple[str, float] | None:
    """Return the obvious largest monthly debit category for a short summary."""

    values: list[tuple[str, float]] = []
    categories = SPENDING_CATEGORIES
    if _finite_number(
        category_summary.get("Income", {}).get("monthly_normalized")
    ):
        categories += ("Income",)
    for category in categories:
        amount = _finite_number(
            category_summary.get(category, {}).get("monthly_normalized")
        )
        label = "Income-labelled debit" if category == "Income" else category
        values.append((label, amount or 0.0))
    largest = max(values, key=lambda item: item[1])
    return largest if largest[1] > 0 else None
