"""In-memory behavioral analytics for confirmed uploaded transactions.

The upload path deliberately does not use SQLite. It adapts reviewed statement
data to the existing FinPulse score, signal, and recommendation functions while
keeping unavailable history-dependent features explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from scripts.recommendation_engine import generate_recommendations
from scripts.score_engine import (
    debt_score,
    savings_score,
    score_band,
    spending_control_score,
    stability_score,
)
from scripts.signal_engine import generate_signals

from .review import (
    FINPULSE_CATEGORIES,
    validate_final_categories,
    validate_include_in_analysis,
)


CATEGORY_AMOUNT_COLUMNS = {
    "Essentials": "essentials",
    "Desire": "desire",
    "Repayment": "repayment",
    "Investment/Savings": "investment_savings",
    "Others": "others",
}

TREND_COLUMNS = (
    "income_change_3m",
    "desire_ratio_change_3m",
    "repayment_ratio_change_3m",
    "investment_ratio_change_3m",
    "expense_ratio_change_3m",
    "surplus_ratio_change_3m",
)


@dataclass
class UploadAnalyticsResult:
    statement_summary: dict[str, Any]
    category_summary: dict[str, dict[str, Any]]
    behavioral_features: dict[str, Any]
    score_result: dict[str, Any]
    signals: list[str]
    recommendations: list[dict[str, Any]]
    data_quality: dict[str, Any]
    budget_summary: dict[str, Any] | None
    observed_income_summary: dict[str, Any]
    monthly_summary: pd.DataFrame


def _positive_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a positive number")
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return number


def _validate_inputs(
    transactions: pd.DataFrame,
    monthly_available_amount: object,
    monthly_budget: object,
) -> tuple[pd.DataFrame, float, float | None, int, int]:
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("reviewed_transactions must be a pandas DataFrame")
    if transactions.empty:
        raise ValueError("reviewed_transactions must contain at least one transaction")

    required = {"date", "amount", "transaction_type", "final_category"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(
            "reviewed_transactions is missing required columns: "
            + ", ".join(sorted(missing))
        )
    prepared = transactions.copy()
    if "include_in_analysis" not in prepared.columns:
        prepared["include_in_analysis"] = True
    validate_final_categories(prepared)
    validate_include_in_analysis(prepared)

    original_transaction_count = len(prepared)
    prepared = prepared.loc[prepared["include_in_analysis"]].copy()
    excluded_transaction_count = original_transaction_count - len(prepared)
    if prepared.empty:
        raise ValueError(
            "No transactions are included in FinPulse analysis. Include at least "
            "one transaction before generating the report."
        )

    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    if prepared["date"].isna().any():
        raise ValueError("reviewed_transactions contains invalid dates")
    prepared["date"] = prepared["date"].dt.normalize()

    prepared["amount"] = pd.to_numeric(prepared["amount"], errors="coerce")
    if prepared["amount"].isna().any() or (prepared["amount"] <= 0).any():
        raise ValueError("reviewed_transactions contains invalid transaction amounts")

    prepared["transaction_type"] = (
        prepared["transaction_type"].astype("string").str.casefold().str.strip()
    )
    invalid_directions = set(prepared["transaction_type"].dropna()) - {"debit", "credit"}
    if invalid_directions or prepared["transaction_type"].isna().any():
        raise ValueError(
            "transaction_type must contain only 'debit' or 'credit'"
        )

    monthly_amount = _positive_number(
        monthly_available_amount,
        "monthly_available_amount",
    )
    if monthly_budget is None or pd.isna(monthly_budget):
        budget = None
    else:
        budget = _positive_number(monthly_budget, "monthly_budget")
    return (
        prepared.sort_values("date").copy(),
        monthly_amount,
        budget,
        original_transaction_count,
        excluded_transaction_count,
    )


def _period_coverage(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    periods = pd.period_range(start=start_date, end=end_date, freq="M")
    for period in periods:
        month_start = period.start_time.normalize()
        month_end = period.end_time.normalize()
        observed_start = max(start_date, month_start)
        observed_end = min(end_date, month_end)
        observed_days = (observed_end - observed_start).days + 1
        days_in_month = period.days_in_month
        rows.append({
            "month": str(period),
            "period": period,
            "coverage_days": observed_days,
            "days_in_month": days_in_month,
            "coverage_fraction": observed_days / days_in_month,
            "is_complete_month": (
                observed_start == month_start and observed_end == month_end
            ),
        })
    return pd.DataFrame(rows)


def _build_monthly_summary(
    transactions: pd.DataFrame,
    coverage: pd.DataFrame,
    monthly_amount: float,
) -> pd.DataFrame:
    working = transactions.copy()
    working["period"] = working["date"].dt.to_period("M")
    rows = []

    for _, coverage_row in coverage.iterrows():
        period = coverage_row["period"]
        month_transactions = working.loc[working["period"] == period]
        debit_rows = month_transactions["transaction_type"] == "debit"
        credit_rows = month_transactions["transaction_type"] == "credit"

        amounts = {}
        for category, column in CATEGORY_AMOUNT_COLUMNS.items():
            amounts[column] = float(
                month_transactions.loc[
                    debit_rows & (month_transactions["final_category"] == category),
                    "amount",
                ].sum()
            )

        total_debit = float(month_transactions.loc[debit_rows, "amount"].sum())
        total_credit = float(month_transactions.loc[credit_rows, "amount"].sum())
        observed_income = float(
            month_transactions.loc[
                credit_rows & (month_transactions["final_category"] == "Income"),
                "amount",
            ].sum()
        )
        expense_ratio = total_debit / monthly_amount
        surplus = monthly_amount - total_debit

        rows.append({
            "month": coverage_row["month"],
            "coverage_days": int(coverage_row["coverage_days"]),
            "days_in_month": int(coverage_row["days_in_month"]),
            "coverage_fraction": float(coverage_row["coverage_fraction"]),
            "is_complete_month": bool(coverage_row["is_complete_month"]),
            "transaction_count": int(len(month_transactions)),
            "credit_transaction_count": int(credit_rows.sum()),
            "debit_transaction_count": int(debit_rows.sum()),
            "observed_income": observed_income,
            "total_credit_inflow": total_credit,
            **amounts,
            "total_debit_spending": total_debit,
            "surplus": surplus,
            "essential_ratio": amounts["essentials"] / monthly_amount,
            "desire_ratio": amounts["desire"] / monthly_amount,
            "repayment_ratio": amounts["repayment"] / monthly_amount,
            "investment_ratio": amounts["investment_savings"] / monthly_amount,
            "other_ratio": amounts["others"] / monthly_amount,
            "expense_to_income_ratio": expense_ratio,
            "surplus_ratio": surplus / monthly_amount,
            "overspending_flag": int(total_debit > monthly_amount),
        })

    summary = pd.DataFrame(rows)
    amount_columns = [
        "observed_income",
        "total_credit_inflow",
        *CATEGORY_AMOUNT_COLUMNS.values(),
        "total_debit_spending",
        "surplus",
    ]
    ratio_columns = [
        "coverage_fraction",
        "essential_ratio",
        "desire_ratio",
        "repayment_ratio",
        "investment_ratio",
        "other_ratio",
        "expense_to_income_ratio",
        "surplus_ratio",
    ]
    summary[amount_columns] = summary[amount_columns].round(2)
    summary[ratio_columns] = summary[ratio_columns].round(4)
    return summary


def _trend_features(complete_months: pd.DataFrame) -> dict[str, float | None]:
    trends = {column: None for column in TREND_COLUMNS}
    if len(complete_months) < 6:
        return trends

    last_six = complete_months.tail(6)
    previous = last_six.iloc[:3]
    recent = last_six.iloc[3:]
    trends.update({
        "income_change_3m": (
            recent["observed_income"].mean() - previous["observed_income"].mean()
        ),
        "desire_ratio_change_3m": (
            recent["desire_ratio"].mean() - previous["desire_ratio"].mean()
        ),
        "repayment_ratio_change_3m": (
            recent["repayment_ratio"].mean() - previous["repayment_ratio"].mean()
        ),
        "investment_ratio_change_3m": (
            recent["investment_ratio"].mean() - previous["investment_ratio"].mean()
        ),
        "expense_ratio_change_3m": (
            recent["expense_to_income_ratio"].mean()
            - previous["expense_to_income_ratio"].mean()
        ),
        "surplus_ratio_change_3m": (
            recent["surplus_ratio"].mean() - previous["surplus_ratio"].mean()
        ),
    })
    return {
        key: None if value is None else round(float(value), 4)
        for key, value in trends.items()
    }


def _behavioral_features(
    transactions: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    monthly_amount: float,
    normalization_months: float,
) -> tuple[dict[str, Any], bool, bool]:
    debit_rows = transactions["transaction_type"] == "debit"
    totals = {
        column: float(
            transactions.loc[
                debit_rows & (transactions["final_category"] == category),
                "amount",
            ].sum()
        )
        for category, column in CATEGORY_AMOUNT_COLUMNS.items()
    }
    total_debit = float(transactions.loc[debit_rows, "amount"].sum())
    denominator = monthly_amount * normalization_months

    complete_months = monthly_summary.loc[
        monthly_summary["is_complete_month"]
    ].copy()
    volatility_available = len(complete_months) >= 2
    expense_volatility = (
        float(complete_months["expense_to_income_ratio"].std())
        if volatility_available
        else None
    )
    desire_volatility = (
        float(complete_months["desire_ratio"].std())
        if volatility_available
        else None
    )
    repayment_volatility = (
        float(complete_months["repayment_ratio"].std())
        if volatility_available
        else None
    )
    investment_volatility = (
        float(complete_months["investment_ratio"].std())
        if volatility_available
        else None
    )

    complete_income = complete_months["observed_income"]
    income_history_available = (
        len(complete_income) >= 2 and bool((complete_income > 0).all())
    )
    income_std = float(complete_income.std()) if income_history_available else None
    income_cv = (
        income_std / float(complete_income.mean())
        if income_history_available and float(complete_income.mean()) > 0
        else None
    )
    stability_available = income_cv is not None and expense_volatility is not None

    overspending_basis = (
        complete_months if not complete_months.empty else monthly_summary
    )
    overspending_months = int(overspending_basis["overspending_flag"].sum())
    overspending_ratio = (
        float(overspending_basis["overspending_flag"].mean())
        if not overspending_basis.empty
        else 0.0
    )

    trends = _trend_features(complete_months)
    trend_available = all(trends[column] is not None for column in TREND_COLUMNS)
    expense_ratio = total_debit / denominator
    surplus = monthly_amount - (total_debit / normalization_months)
    surplus_ratio = 1.0 - expense_ratio

    features: dict[str, Any] = {
        "user_id": "UPLOADED_USER",
        "months_observed": int(len(monthly_summary)),
        "complete_months_observed": int(len(complete_months)),
        "monthly_reference_amount": monthly_amount,
        "avg_income": monthly_amount,
        "avg_income_source": "declared_monthly_reference_not_observed_salary",
        "avg_essential_ratio": totals["essentials"] / denominator,
        "avg_desire_ratio": totals["desire"] / denominator,
        "avg_repayment_ratio": totals["repayment"] / denominator,
        "avg_investment_ratio": totals["investment_savings"] / denominator,
        "avg_other_ratio": totals["others"] / denominator,
        "avg_expense_ratio": expense_ratio,
        "avg_surplus": surplus,
        "avg_surplus_ratio": surplus_ratio,
        "avg_remaining_ratio": surplus_ratio,
        "avg_monthly_transactions": len(transactions) / normalization_months,
        "overspending_months": overspending_months,
        "overspending_month_ratio": overspending_ratio,
        "income_std": income_std,
        "income_cv": income_cv,
        "expense_ratio_volatility": expense_volatility,
        "desire_volatility": desire_volatility,
        "repayment_volatility": repayment_volatility,
        "investment_volatility": investment_volatility,
        **trends,
    }
    for key, value in features.items():
        if isinstance(value, (float, np.floating)) and np.isfinite(value):
            features[key] = round(float(value), 4)
    return features, stability_available, trend_available


def _legacy_adapter(features: dict[str, Any]) -> dict[str, Any]:
    """Use neutral values only to suppress unavailable legacy history rules."""

    adapted = dict(features)
    adapted["income_cv"] = features["income_cv"] or 0.0
    adapted["expense_ratio_volatility"] = (
        features["expense_ratio_volatility"] or 0.0
    )
    for column in TREND_COLUMNS:
        adapted[column] = features[column] if features[column] is not None else 0.0
    return adapted


def _score_upload(
    features: dict[str, Any],
    stability_available: bool,
    trend_available: bool,
    analytical_confidence: str,
) -> dict[str, Any]:
    adapted = _legacy_adapter(features)
    spending = int(spending_control_score(adapted))
    savings = int(savings_score(adapted))
    debt = int(debt_score(adapted))
    stability = int(stability_score(adapted)) if stability_available else None

    components = {
        "spending_control": {
            "score": spending,
            "maximum": 30,
            "available": True,
            "inputs": {
                "avg_expense_ratio": features["avg_expense_ratio"],
                "overspending_month_ratio": features["overspending_month_ratio"],
            },
        },
        "savings": {
            "score": savings,
            "maximum": 25,
            "available": True,
            "inputs": {
                "avg_investment_ratio": features["avg_investment_ratio"],
                "investment_ratio_change_3m": features[
                    "investment_ratio_change_3m"
                ],
            },
            "trend_adjustment_applied": trend_available,
        },
        "debt_management": {
            "score": debt,
            "maximum": 25,
            "available": True,
            "inputs": {
                "avg_repayment_ratio": features["avg_repayment_ratio"],
                "repayment_ratio_change_3m": features[
                    "repayment_ratio_change_3m"
                ],
            },
            "trend_adjustment_applied": trend_available,
        },
        "stability": {
            "score": stability,
            "maximum": 20,
            "available": stability_available,
            "inputs": {
                "income_cv": features["income_cv"],
                "expense_ratio_volatility": features[
                    "expense_ratio_volatility"
                ],
                "avg_surplus_ratio": features["avg_surplus_ratio"],
            },
            "unavailable_reason": (
                None
                if stability_available
                else (
                    "Requires at least two complete calendar months with "
                    "positive observed Income credits."
                )
            ),
        },
    }

    raw_available = spending + savings + debt + (stability or 0)
    available_maximum = 100 if stability_available else 80
    normalized_score = int(round(raw_available / available_maximum * 100))
    provisional = (
        not stability_available
        or not trend_available
        or analytical_confidence != "High"
    )
    return {
        "finpulse_score": normalized_score,
        "score_band": score_band(normalized_score),
        "raw_available_score": raw_available,
        "available_maximum": available_maximum,
        "is_provisional": provisional,
        "renormalized_for_unavailable_components": not stability_available,
        "components": components,
        "method": (
            "Existing FinPulse component rules. When stability is unavailable, "
            "the available 80 points are renormalized to 100; unavailable trend "
            "adjustments are neutral and explicitly marked."
        ),
    }


def _analytical_quality(
    transactions: pd.DataFrame,
    coverage_days: int,
    complete_months: int,
    stability_available: bool,
    trend_available: bool,
    original_transaction_count: int,
    excluded_transaction_count: int,
) -> dict[str, Any]:
    warnings = []
    if len(transactions) < 30:
        warnings.append("Fewer than 30 transactions are available.")
    if coverage_days < 30:
        warnings.append("Statement coverage is shorter than 30 days.")
    if not stability_available:
        warnings.append("The Stability component is unavailable from this history.")
    if not trend_available:
        warnings.append("Three-month trend claims are unavailable.")

    if len(transactions) < 30 or coverage_days < 30:
        confidence = "Low"
    elif (
        len(transactions) >= 90
        and coverage_days >= 180
        and complete_months >= 6
        and stability_available
        and trend_available
    ):
        confidence = "High"
    else:
        confidence = "Medium"

    categorization_counts = (
        transactions["confidence"].value_counts().to_dict()
        if "confidence" in transactions.columns
        else {}
    )
    return {
        "analytical_confidence": confidence,
        "transaction_count": len(transactions),
        "original_reviewed_transaction_count": original_transaction_count,
        "included_transaction_count": len(transactions),
        "excluded_transaction_count": excluded_transaction_count,
        "coverage_days": coverage_days,
        "represented_months": int(
            transactions["date"].dt.to_period("M").nunique()
        ),
        "complete_months": complete_months,
        "minimum_transaction_guidance_met": len(transactions) >= 30,
        "minimum_coverage_guidance_met": coverage_days >= 30,
        "stability_features_available": stability_available,
        "trend_features_available": trend_available,
        "warnings": warnings,
        "categorization_confidence_counts": categorization_counts,
        "note": (
            "Analytical confidence describes behavioral coverage and is separate "
            "from transaction categorization confidence."
        ),
    }


def analyze_reviewed_transactions(
    reviewed_transactions: pd.DataFrame,
    monthly_available_amount: float,
    monthly_budget: float | None = None,
) -> UploadAnalyticsResult:
    """Build upload-specific features and legacy rule outputs entirely in memory."""

    (
        transactions,
        monthly_amount,
        budget,
        original_transaction_count,
        excluded_transaction_count,
    ) = _validate_inputs(
        reviewed_transactions,
        monthly_available_amount,
        monthly_budget,
    )
    start_date = transactions["date"].min()
    end_date = transactions["date"].max()
    coverage_days = int((end_date - start_date).days + 1)
    coverage = _period_coverage(start_date, end_date)
    covered_month_equivalents = float(coverage["coverage_fraction"].sum())
    normalization_months = max(1.0, covered_month_equivalents)

    monthly_summary = _build_monthly_summary(
        transactions,
        coverage,
        monthly_amount,
    )
    features, stability_available, trend_available = _behavioral_features(
        transactions,
        monthly_summary,
        monthly_amount,
        normalization_months,
    )
    complete_month_count = int(monthly_summary["is_complete_month"].sum())
    data_quality = _analytical_quality(
        transactions,
        coverage_days,
        complete_month_count,
        stability_available,
        trend_available,
        original_transaction_count,
        excluded_transaction_count,
    )

    debit_rows = transactions["transaction_type"] == "debit"
    credit_rows = transactions["transaction_type"] == "credit"
    category_summary: dict[str, dict[str, Any]] = {}
    for category in FINPULSE_CATEGORIES:
        direction_mask = credit_rows if category == "Income" else debit_rows
        category_rows = transactions.loc[
            direction_mask & (transactions["final_category"] == category)
        ]
        total = float(category_rows["amount"].sum())
        category_summary[category] = {
            "total": round(total, 2),
            "monthly_normalized": round(total / normalization_months, 2),
            "transaction_count": int(len(category_rows)),
        }

    total_debit = float(transactions.loc[debit_rows, "amount"].sum())
    total_credit = float(transactions.loc[credit_rows, "amount"].sum())
    observed_income = category_summary["Income"]["total"]
    monthly_debit = total_debit / normalization_months
    monthly_credit = total_credit / normalization_months

    budget_summary = None
    if budget is not None:
        budget_summary = {
            "monthly_budget": round(budget, 2),
            "monthly_normalized_debit_spending": round(monthly_debit, 2),
            "budget_utilization": round(monthly_debit / budget, 4),
            "budget_utilization_percent": round(monthly_debit / budget * 100, 2),
            "budget_variance": round(budget - monthly_debit, 2),
            "over_budget": monthly_debit > budget,
        }

    statement_summary = {
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "coverage_days": coverage_days,
        "transaction_count": len(transactions),
        "original_reviewed_transaction_count": original_transaction_count,
        "included_transaction_count": len(transactions),
        "excluded_transaction_count": excluded_transaction_count,
        "covered_calendar_months": monthly_summary["month"].tolist(),
        "covered_calendar_month_count": len(monthly_summary),
        "transaction_month_count": int(
            transactions["date"].dt.to_period("M").nunique()
        ),
        "complete_calendar_months": monthly_summary.loc[
            monthly_summary["is_complete_month"], "month"
        ].tolist(),
        "covered_month_equivalents": round(covered_month_equivalents, 4),
        "normalization_months": round(normalization_months, 4),
        "normalization_method": (
            "Sum observed-days/days-in-calendar-month across the statement. "
            "Use a one-month floor for histories shorter than one month to avoid "
            "extrapolating partial-period spending upward."
        ),
        "coverage_boundary_assumption": (
            "Coverage starts at the earliest transaction date and ends at the "
            "latest transaction date because statement-period metadata is unavailable."
        ),
        "monthly_available_amount": round(monthly_amount, 2),
        "total_debit_spending": round(total_debit, 2),
        "monthly_normalized_debit_spending": round(monthly_debit, 2),
        "remaining_amount_estimate": round(monthly_amount - monthly_debit, 2),
    }
    observed_income_summary = {
        "observed_income_credits": round(float(observed_income), 2),
        "monthly_normalized_observed_income": round(
            float(observed_income) / normalization_months, 2
        ),
        "total_credit_inflow": round(total_credit, 2),
        "monthly_normalized_credit_inflow": round(monthly_credit, 2),
        "monthly_reference_amount": round(monthly_amount, 2),
        "reference_amount_was_replaced_by_credits": False,
    }

    score_result = _score_upload(
        features,
        stability_available,
        trend_available,
        data_quality["analytical_confidence"],
    )
    legacy_features = _legacy_adapter(features)
    signals = generate_signals(pd.Series(legacy_features))
    recommendations = generate_recommendations(pd.Series(legacy_features))[:3]

    return UploadAnalyticsResult(
        statement_summary=statement_summary,
        category_summary=category_summary,
        behavioral_features=features,
        score_result=score_result,
        signals=signals,
        recommendations=recommendations,
        data_quality=data_quality,
        budget_summary=budget_summary,
        observed_income_summary=observed_income_summary,
        monthly_summary=monthly_summary,
    )
