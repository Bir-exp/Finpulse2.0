"""In-memory orchestration and view data for the uploaded-user report.

The Streamlit page renders this module's structured output.  Financial
calculations remain in ``upload_analytics`` and persona inference remains in
``segmentation_inference``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Real
from typing import Any, MutableMapping

import numpy as np
import pandas as pd

from .review import FINPULSE_CATEGORIES, initialize_review_categories
from .segmentation_inference import PersonaPrediction, predict_persona
from .upload_analytics import UploadAnalyticsResult, analyze_reviewed_transactions


REPORT_KEY = "finpulse_upload_report"
ANALYTICS_KEY = "upload_analytics_result"
PERSONA_KEY = "upload_persona_result"
REPORT_SIGNATURE_KEY = "finpulse_report_signature"
REPORT_STATE_KEYS = (
    REPORT_KEY,
    ANALYTICS_KEY,
    PERSONA_KEY,
    REPORT_SIGNATURE_KEY,
)

STATEMENT_DERIVED_STATE_KEYS = (
    "categorized_transactions",
    "reviewed_transactions",
    "statement_diagnostics",
    "statement_rejected_rows",
    "statement_source_name",
    "statement_upload_bytes",
    "manual_mapping_required",
    "review_filter_mode",
)

FILE_FINGERPRINT_KEY = "statement_file_fingerprint"
MONTHLY_AMOUNT_KEY = "monthly_available_amount"
MONTHLY_BUDGET_KEY = "monthly_budget"


class ReportNotReadyError(ValueError):
    """Raised when report generation is attempted before review confirmation."""


@dataclass(frozen=True)
class ScorePresentation:
    is_provisional: bool
    status_label: str
    explanation: str


@dataclass(frozen=True)
class FinPulseReport:
    analytics: UploadAnalyticsResult
    persona: PersonaPrediction
    category_breakdown: pd.DataFrame
    debit_spending_breakdown: pd.DataFrame
    behavioral_ratios: dict[str, float | None]
    score_presentation: ScorePresentation
    review_summary: dict[str, int]
    excluded_transactions: pd.DataFrame


def _same_number(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, Real) or not isinstance(right, Real):
        return False
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-9))


def invalidate_report_state(state: MutableMapping[str, Any]) -> tuple[str, ...]:
    """Remove only generated analytics/persona/report objects from a session."""

    removed = []
    for key in REPORT_STATE_KEYS:
        if key in state:
            state.pop(key, None)
            removed.append(key)
    return tuple(removed)


def invalidate_review_confirmation(
    state: MutableMapping[str, Any],
) -> tuple[str, ...]:
    """Invalidate confirmation and every result derived from review decisions."""

    removed = list(invalidate_report_state(state))
    if "reviewed_transactions" in state:
        state.pop("reviewed_transactions", None)
        removed.append("reviewed_transactions")
    return tuple(removed)


def synchronize_statement_inputs(
    state: MutableMapping[str, Any],
    *,
    file_fingerprint: str,
    monthly_available_amount: float,
    monthly_budget: float | None,
) -> tuple[str, ...]:
    """Invalidate stale session results when report inputs change.

    A different file invalidates ingestion, review, and report state. Monetary
    input changes preserve the reviewed rows but invalidate analytics and the
    report because categorization is independent of those values.
    """

    changes: list[str] = []
    previous_fingerprint = state.get(FILE_FINGERPRINT_KEY)
    file_changed = (
        previous_fingerprint is not None
        and previous_fingerprint != file_fingerprint
    )
    amount_changed = (
        MONTHLY_AMOUNT_KEY in state
        and not _same_number(
            state.get(MONTHLY_AMOUNT_KEY), monthly_available_amount
        )
    )
    budget_changed = (
        MONTHLY_BUDGET_KEY in state
        and not _same_number(state.get(MONTHLY_BUDGET_KEY), monthly_budget)
    )

    if file_changed:
        for key in (*STATEMENT_DERIVED_STATE_KEYS, *REPORT_STATE_KEYS):
            state.pop(key, None)
        changes.append("uploaded_file")
    else:
        if amount_changed:
            changes.append("monthly_available_amount")
        if budget_changed:
            changes.append("monthly_budget")
        if amount_changed or budget_changed:
            invalidate_report_state(state)

    state[FILE_FINGERPRINT_KEY] = file_fingerprint
    state[MONTHLY_AMOUNT_KEY] = float(monthly_available_amount)
    state[MONTHLY_BUDGET_KEY] = (
        None if monthly_budget is None else float(monthly_budget)
    )
    return tuple(changes)


def reviewed_transactions_fingerprint(transactions: pd.DataFrame) -> str:
    """Return a stable session signature without persisting transaction data."""

    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("reviewed_transactions must be a pandas DataFrame")
    transactions = initialize_review_categories(transactions)
    relevant = [
        column
        for column in (
            "date",
            "amount",
            "transaction_type",
            "predicted_category",
            "final_category",
            "include_in_analysis",
        )
        if column in transactions.columns
    ]
    row_hashes = pd.util.hash_pandas_object(
        transactions[relevant], index=True
    ).to_numpy(dtype="uint64")
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def build_report_signature(
    *,
    file_fingerprint: str,
    reviewed_transactions: pd.DataFrame,
    monthly_available_amount: float,
    monthly_budget: float | None,
) -> str:
    transaction_hash = reviewed_transactions_fingerprint(reviewed_transactions)
    budget_text = "none" if monthly_budget is None else f"{float(monthly_budget):.12g}"
    payload = "|".join(
        (
            file_fingerprint,
            transaction_hash,
            f"{float(monthly_available_amount):.12g}",
            budget_text,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_score_presentation(score_result: dict[str, Any]) -> ScorePresentation:
    provisional = bool(score_result.get("is_provisional", False))
    stability = score_result.get("components", {}).get("stability", {})
    stability_available = bool(stability.get("available", False))
    renormalized = bool(
        score_result.get("renormalized_for_unavailable_components", False)
    )

    if provisional and not stability_available and renormalized:
        explanation = (
            "Stability is unavailable for this history. The available 80 points "
            "were normalized to 100, so this score is provisional and is not "
            "equivalent to a full-history score."
        )
    elif provisional:
        explanation = (
            "This score is provisional because some history-dependent evidence "
            "or overall report confidence is limited."
        )
    else:
        explanation = (
            "All four score components are available from the uploaded history."
        )
    return ScorePresentation(
        is_provisional=provisional,
        status_label="PROVISIONAL" if provisional else "FULL-HISTORY",
        explanation=explanation,
    )


def _category_breakdown(analytics: UploadAnalyticsResult) -> pd.DataFrame:
    rows = []
    for category in FINPULSE_CATEGORIES:
        summary = analytics.category_summary[category]
        rows.append(
            {
                "category": category,
                "total": float(summary["total"]),
                "monthly_normalized": float(summary["monthly_normalized"]),
                "transaction_count": int(summary["transaction_count"]),
                "cash_flow": "credit" if category == "Income" else "debit",
            }
        )
    return pd.DataFrame(rows)


def generate_financial_report(
    reviewed_transactions: pd.DataFrame | None,
    monthly_available_amount: float,
    monthly_budget: float | None = None,
    *,
    review_confirmed: bool,
) -> FinPulseReport:
    """Generate the complete report from confirmed, session-only reviewed rows."""

    if not review_confirmed:
        raise ReportNotReadyError(
            "Confirm reviewed transactions before generating the FinPulse report"
        )
    if not isinstance(reviewed_transactions, pd.DataFrame) or reviewed_transactions.empty:
        raise ReportNotReadyError(
            "Confirmed reviewed transactions are unavailable for report generation"
        )

    normalized_review = initialize_review_categories(reviewed_transactions)
    analytics = analyze_reviewed_transactions(
        normalized_review,
        monthly_available_amount,
        monthly_budget,
    )
    persona = predict_persona(analytics)
    category_breakdown = _category_breakdown(analytics)
    debit_breakdown = category_breakdown.loc[
        category_breakdown["cash_flow"] == "debit"
    ].reset_index(drop=True)
    features = analytics.behavioral_features
    ratios = {
        "Essential": features.get("avg_essential_ratio"),
        "Desire": features.get("avg_desire_ratio"),
        "Repayment": features.get("avg_repayment_ratio"),
        "Investment/Savings": features.get("avg_investment_ratio"),
        "Expense": features.get("avg_expense_ratio"),
        "Surplus/Remaining": features.get("avg_surplus_ratio"),
    }
    low_confidence = (
        int((normalized_review["confidence"] == "Low").sum())
        if "confidence" in normalized_review.columns
        else 0
    )
    manually_corrected = (
        int(
            (
                normalized_review["final_category"]
                != normalized_review["predicted_category"]
            ).sum()
        )
        if "predicted_category" in normalized_review.columns
        else 0
    )
    excluded_transactions = normalized_review.loc[
        ~normalized_review["include_in_analysis"]
    ].copy()
    return FinPulseReport(
        analytics=analytics,
        persona=persona,
        category_breakdown=category_breakdown,
        debit_spending_breakdown=debit_breakdown,
        behavioral_ratios=ratios,
        score_presentation=build_score_presentation(analytics.score_result),
        review_summary={
            "transaction_count": len(normalized_review),
            "included_in_analysis": int(
                normalized_review["include_in_analysis"].sum()
            ),
            "excluded_from_analysis": len(excluded_transactions),
            "low_confidence_automatic_classifications": low_confidence,
            "manually_corrected": manually_corrected,
        },
        excluded_transactions=excluded_transactions,
    )
