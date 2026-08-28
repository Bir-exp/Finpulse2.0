"""Pure presentation helpers for the FinPulse Streamlit interface."""

from __future__ import annotations

from datetime import date
import math
from numbers import Real
from typing import Any, Mapping

import pandas as pd


SPENDING_CATEGORIES = (
    "Essentials",
    "Desire",
    "Repayment",
    "Investment/Savings",
    "Others",
)

MONTHLY_CATEGORY_COLUMNS = {
    "Essentials": "essentials",
    "Desire": "desire",
    "Repayment": "repayment",
    "Investment/Savings": "investment_savings",
    "Others": "others",
}
OTHER_INCLUDED_DEBITS = "Other included debits"


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


def format_month_label(value: object, *, abbreviated: bool = False) -> str:
    """Format a monthly period for selectors or compact chart axes."""

    try:
        period = value if isinstance(value, pd.Period) else pd.Period(str(value), freq="M")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid monthly period: {value!r}") from error
    return period.strftime("%b %Y" if abbreviated else "%B %Y")


def build_monthly_spending_view(
    monthly_summary: pd.DataFrame,
    monthly_budget: object = None,
) -> pd.DataFrame:
    """Project existing upload monthly analytics into a UI-safe spending view.

    Values remain actual included-debit amounts observed in each month. Partial
    months are labelled and are never extrapolated here.
    """

    required = {
        "month",
        "coverage_days",
        "days_in_month",
        "is_complete_month",
        "debit_transaction_count",
        "income_category_debits",
        "total_debit_spending",
        *MONTHLY_CATEGORY_COLUMNS.values(),
    }
    if not isinstance(monthly_summary, pd.DataFrame):
        raise TypeError("monthly_summary must be a pandas DataFrame")
    missing = required - set(monthly_summary.columns)
    if missing:
        raise ValueError(
            "monthly_summary is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if monthly_summary.empty:
        return pd.DataFrame(
            columns=(
                "month_key",
                "month_period",
                "Month",
                "Axis Month",
                "Month Selector",
                "Partial Month",
                "Coverage Days",
                "Days in Month",
                "Transactions",
                "Monthly Spending",
                *MONTHLY_CATEGORY_COLUMNS,
                OTHER_INCLUDED_DEBITS,
                "Budget Used",
            )
        )

    budget = _finite_number(monthly_budget)
    if monthly_budget is not None and (budget is None or budget <= 0):
        raise ValueError("monthly_budget must be a positive finite number or None")

    source = monthly_summary.copy()
    try:
        source["month_period"] = pd.PeriodIndex(source["month"], freq="M")
    except (TypeError, ValueError) as error:
        raise ValueError("monthly_summary contains an invalid month") from error
    source = source.sort_values("month_period", kind="stable").reset_index(drop=True)
    multiple_years = source["month_period"].map(lambda period: period.year).nunique() > 1

    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        period = row["month_period"]
        partial = not bool(row["is_complete_month"])
        axis_label = period.strftime("%b %Y" if multiple_years else "%b")
        if partial:
            axis_label += "*"
        month_label = format_month_label(period)
        category_values = {
            label: float(row[column])
            for label, column in MONTHLY_CATEGORY_COLUMNS.items()
        }
        total = float(row["total_debit_spending"])
        other_included = float(row["income_category_debits"])
        reconciled = sum(category_values.values()) + other_included
        if not math.isclose(reconciled, total, rel_tol=0.0, abs_tol=0.01):
            raise ValueError(
                f"Monthly category totals do not reconcile for {period}: "
                f"categories={reconciled}, spending={total}"
            )
        rows.append(
            {
                "month_key": str(period),
                "month_period": period,
                "Month": month_label,
                "Axis Month": axis_label,
                "Month Selector": month_label + (" (partial)" if partial else ""),
                "Partial Month": partial,
                "Coverage Days": int(row["coverage_days"]),
                "Days in Month": int(row["days_in_month"]),
                "Transactions": int(row["debit_transaction_count"]),
                "Monthly Spending": round(total, 2),
                **{key: round(value, 2) for key, value in category_values.items()},
                OTHER_INCLUDED_DEBITS: round(other_included, 2),
                "Budget Used": None if budget is None else round(total / budget, 4),
            }
        )
    return pd.DataFrame(rows)


def monthly_spending_snapshot(
    monthly_view: pd.DataFrame,
    selection: str,
) -> dict[str, Any]:
    """Return one exact observed-month snapshot selected by key or UI label."""

    if not isinstance(monthly_view, pd.DataFrame) or monthly_view.empty:
        raise ValueError("monthly_view must contain at least one month")
    matches = monthly_view.loc[
        (monthly_view["month_key"] == selection)
        | (monthly_view["Month Selector"] == selection)
    ]
    if len(matches) != 1:
        raise KeyError(f"Unknown or ambiguous month selection: {selection!r}")
    return matches.iloc[0].to_dict()


def monthly_spending_observations(
    monthly_view: pd.DataFrame,
) -> tuple[str, ...]:
    """Create up to three direct, deterministic observations from monthly totals."""

    if not isinstance(monthly_view, pd.DataFrame) or len(monthly_view) < 2:
        return ()
    ordered = monthly_view.sort_values("month_period", kind="stable")
    spending = ordered["Monthly Spending"].astype(float)
    observations: list[str] = []

    highest_position = spending.idxmax()
    lowest_position = spending.idxmin()
    highest = ordered.loc[highest_position]
    lowest = ordered.loc[lowest_position]
    if not math.isclose(
        float(highest["Monthly Spending"]),
        float(lowest["Monthly Spending"]),
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        observations.append(
            f"{highest['Month']} had your highest spending at "
            f"{format_inr(highest['Monthly Spending'])}."
        )
        observations.append(
            f"{lowest['Month']} had your lowest spending at "
            f"{format_inr(lowest['Monthly Spending'])}."
        )
    else:
        observations.append(
            f"Included spending was the same across all {len(ordered)} months."
        )

    positive = ordered.loc[ordered["Monthly Spending"] > 0]
    if not positive.empty:
        observation_categories = list(MONTHLY_CATEGORY_COLUMNS)
        if float(positive[OTHER_INCLUDED_DEBITS].sum()) > 0:
            observation_categories.append(OTHER_INCLUDED_DEBITS)
        largest_by_month = positive[observation_categories].idxmax(axis=1)
        category = str(largest_by_month.value_counts().idxmax())
        count = int((largest_by_month == category).sum())
        verb = "were" if category == "Essentials" else "was"
        observations.append(
            f"{category} {verb} your largest category in {count} of "
            f"{len(positive)} months."
        )
    return tuple(observations[:3])
