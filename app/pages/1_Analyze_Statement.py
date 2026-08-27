"""Session-only bank statement ingestion, categorization, and review page."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finpulse.categorization import categorize_transactions  # noqa: E402
from finpulse.review import (  # noqa: E402
    FINPULSE_CATEGORIES,
    apply_category_edits,
    filter_review_transactions,
    initialize_review_categories,
    validate_final_categories,
)
from finpulse.statement_ingestion import StatementReadError, read_statement  # noqa: E402


WORKING_TRANSACTIONS_KEY = "categorized_transactions"
REVIEWED_TRANSACTIONS_KEY = "reviewed_transactions"
MONTHLY_AMOUNT_KEY = "monthly_available_amount"
MONTHLY_BUDGET_KEY = "monthly_budget"
DIAGNOSTICS_KEY = "statement_diagnostics"

ANALYSIS_STATE_KEYS = (
    WORKING_TRANSACTIONS_KEY,
    REVIEWED_TRANSACTIONS_KEY,
    DIAGNOSTICS_KEY,
    "statement_rejected_rows",
    "statement_source_name",
    "statement_file_fingerprint",
)


def _clear_previous_analysis() -> None:
    for key in ANALYSIS_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("review_filter_mode", None)
    st.session_state["review_editor_revision"] = (
        st.session_state.get("review_editor_revision", 0) + 1
    )


def _advance_editor_revision() -> None:
    st.session_state["review_editor_revision"] = (
        st.session_state.get("review_editor_revision", 0) + 1
    )


def _file_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _render_diagnostics(diagnostics: dict[str, object]) -> None:
    st.subheader("Ingestion Summary")

    summary_columns = st.columns(5)
    summary_columns[0].metric("Source rows", diagnostics["original_row_count"])
    summary_columns[1].metric("Usable rows", diagnostics["cleaned_row_count"])
    summary_columns[2].metric("Dropped rows", diagnostics["dropped_row_count"])
    summary_columns[3].metric("Debits", diagnostics["debit_transactions"])
    summary_columns[4].metric("Credits", diagnostics["credit_transactions"])

    start_date, end_date = diagnostics["date_range"]
    if start_date and end_date:
        st.caption(f"Detected statement range: {start_date} to {end_date}")

    missing = diagnostics["missing_required_fields"]
    if missing:
        st.error("Missing required mapping: " + ", ".join(missing))

    ambiguities = diagnostics["ambiguous_mappings"]
    if ambiguities:
        st.warning(
            "Some source columns are ambiguous and need explicit mapping in a future "
            "manual-mapping step."
        )

    with st.expander("Detected column mapping and validation details"):
        st.write("Detected mapping")
        st.json(diagnostics["detected_column_mapping"])
        if ambiguities:
            st.write("Ambiguous candidates")
            st.json(ambiguities)
        st.write("All mapping candidates")
        st.json(diagnostics["candidate_columns"])
        if diagnostics["dropped_row_reasons"]:
            st.write("Dropped-row reasons")
            st.json(diagnostics["dropped_row_reasons"])

    rejected = st.session_state.get("statement_rejected_rows")
    if isinstance(rejected, pd.DataFrame) and not rejected.empty:
        with st.expander("Inspect rows excluded during cleaning"):
            st.dataframe(rejected, use_container_width=True)


def _statement_coverage(transactions: pd.DataFrame) -> tuple[int, str, int]:
    transaction_count = len(transactions)
    parsed_dates = pd.to_datetime(transactions["date"], errors="coerce").dropna()
    if parsed_dates.empty:
        return transaction_count, "Unavailable", 0

    start = parsed_dates.min().date()
    end = parsed_dates.max().date()
    coverage_days = (end - start).days + 1
    return transaction_count, f"{start.isoformat()} to {end.isoformat()}", coverage_days


st.set_page_config(
    page_title="Analyze Statement | FinPulse",
    page_icon="📄",
    layout="wide",
)

st.title("Analyze Your Bank Statement")
st.caption(
    "Upload and review transactions privately in this session. Nothing on this "
    "page writes statement data to SQLite or disk."
)

with st.form("statement_setup_form"):
    uploaded_file = st.file_uploader(
        "Upload Statement",
        type=["csv", "xlsx"],
        help="CSV and XLSX statements are supported. PDF support will be added later.",
    )

    monthly_available_amount = st.number_input(
        "Monthly Available Amount",
        min_value=0.0,
        value=st.session_state.get(MONTHLY_AMOUNT_KEY),
        step=500.0,
        help=(
            "Your salary, stipend, pocket money, household allowance, or other "
            "monthly funds available."
        ),
    )

    monthly_budget = st.number_input(
        "Monthly Spending Budget (optional)",
        min_value=0.0,
        value=st.session_state.get(MONTHLY_BUDGET_KEY),
        step=500.0,
        help="Leave this blank if you do not use a monthly spending budget.",
    )

    analyze_statement = st.form_submit_button(
        "Analyze Statement",
        type="primary",
    )


if analyze_statement:
    if uploaded_file is None:
        st.error("Upload a CSV or XLSX bank statement before analyzing.")
    elif monthly_available_amount is None or monthly_available_amount <= 0:
        st.error("Monthly Available Amount is required and must be greater than zero.")
    elif monthly_budget is not None and monthly_budget <= 0:
        st.error("Monthly Spending Budget must be greater than zero or left blank.")
    else:
        statement_bytes = uploaded_file.getvalue()
        _clear_previous_analysis()
        st.session_state[MONTHLY_AMOUNT_KEY] = float(monthly_available_amount)
        st.session_state[MONTHLY_BUDGET_KEY] = (
            None if monthly_budget is None else float(monthly_budget)
        )
        st.session_state["statement_source_name"] = uploaded_file.name
        st.session_state["statement_file_fingerprint"] = _file_fingerprint(
            statement_bytes
        )

        try:
            ingestion_result = read_statement(
                statement_bytes,
                filename=uploaded_file.name,
            )
            st.session_state[DIAGNOSTICS_KEY] = asdict(
                ingestion_result.diagnostics
            )
            st.session_state["statement_rejected_rows"] = (
                ingestion_result.rejected_rows.copy()
            )

            if ingestion_result.is_usable:
                categorized = categorize_transactions(
                    ingestion_result.transactions
                )
                st.session_state[WORKING_TRANSACTIONS_KEY] = (
                    initialize_review_categories(categorized)
                )
            else:
                st.error(
                    "The statement could not be standardized automatically. "
                    "Review the mapping diagnostics below."
                )
        except (StatementReadError, ValueError) as error:
            st.error(str(error))


diagnostics = st.session_state.get(DIAGNOSTICS_KEY)
if diagnostics:
    _render_diagnostics(diagnostics)


working_transactions = st.session_state.get(WORKING_TRANSACTIONS_KEY)
if isinstance(working_transactions, pd.DataFrame) and not working_transactions.empty:
    st.divider()
    st.subheader("Statement Coverage")

    transaction_count, date_range_text, coverage_days = _statement_coverage(
        working_transactions
    )
    coverage_columns = st.columns(3)
    coverage_columns[0].metric("Transactions", transaction_count)
    coverage_columns[1].metric("Approximate coverage", f"{coverage_days} days")
    coverage_columns[2].metric("Low-confidence rows", int(
        (working_transactions["confidence"] == "Low").sum()
    ))
    st.caption(f"Transaction date range: {date_range_text}")

    limited_reasons = []
    if transaction_count < 30:
        limited_reasons.append("fewer than 30 transactions")
    if coverage_days < 30:
        limited_reasons.append("fewer than 30 days of coverage")
    if limited_reasons:
        st.warning(
            "Limited statement history ("
            + " and ".join(limited_reasons)
            + "). You can continue, but a later behavioral profile will have "
            "lower confidence."
        )
    else:
        st.success("The statement meets the recommended 30-day and 30-transaction coverage.")

    st.divider()
    st.subheader("Categorization Summary")
    confidence_counts = working_transactions["confidence"].value_counts()
    confidence_columns = st.columns(3)
    for position, confidence in enumerate(("High", "Medium", "Low")):
        confidence_columns[position].metric(
            f"{confidence} confidence",
            int(confidence_counts.get(confidence, 0)),
        )

    st.divider()
    st.subheader("Review Transactions")
    st.caption(
        "Predicted categories remain unchanged. Edit Final Category where a "
        "correction is needed, then explicitly confirm the reviewed data."
    )

    st.session_state.setdefault("review_editor_revision", 0)
    review_mode = st.radio(
        "Rows to show",
        options=("All transactions", "Needs review (Low confidence)"),
        horizontal=True,
        key="review_filter_mode",
        on_change=_advance_editor_revision,
    )
    low_only = review_mode == "Needs review (Low confidence)"
    review_view = filter_review_transactions(
        working_transactions,
        low_confidence_only=low_only,
    )

    if low_only and review_view.empty:
        st.info(
            "There are no Low-confidence transactions. Switch to All transactions "
            "to inspect or edit Medium- and High-confidence rows."
        )
    else:
        display_columns = [
            "date",
            "receiver",
            "description",
            "amount",
            "transaction_type",
            "detailed_category",
            "predicted_category",
            "confidence",
            "final_category",
        ]
        display_columns = [
            column for column in display_columns if column in review_view.columns
        ]
        editor_key = (
            f"statement_review_editor_{review_mode}_"
            f"{st.session_state['review_editor_revision']}"
        )
        edited_view = st.data_editor(
            review_view[display_columns],
            key=editor_key,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=[
                column for column in display_columns if column != "final_category"
            ],
            column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "amount": st.column_config.NumberColumn("Amount", format="₹ %.2f"),
                "final_category": st.column_config.SelectboxColumn(
                    "Final Category",
                    options=list(FINPULSE_CATEGORIES),
                    required=True,
                    help="This reviewed category will be used by later analytics.",
                ),
            },
        )

        try:
            updated_transactions = apply_category_edits(
                working_transactions,
                edited_view,
            )
            categories_changed = not updated_transactions["final_category"].equals(
                working_transactions["final_category"]
            )
            st.session_state[WORKING_TRANSACTIONS_KEY] = updated_transactions
            working_transactions = updated_transactions
            if categories_changed:
                st.session_state.pop(REVIEWED_TRANSACTIONS_KEY, None)
        except (KeyError, TypeError, ValueError) as error:
            st.error(f"Could not apply category edits: {error}")

    st.markdown("### Confirm Reviewed Transactions")
    st.caption(
        "Confirmation creates the session-only reviewed dataset for later phases. "
        "Changing a category afterward requires confirmation again."
    )
    if st.button("Confirm Reviewed Transactions", type="primary"):
        try:
            validate_final_categories(
                st.session_state[WORKING_TRANSACTIONS_KEY]
            )
            st.session_state[REVIEWED_TRANSACTIONS_KEY] = (
                st.session_state[WORKING_TRANSACTIONS_KEY].copy(deep=True)
            )
            st.session_state[MONTHLY_AMOUNT_KEY] = float(monthly_available_amount)
            st.session_state[MONTHLY_BUDGET_KEY] = (
                None if monthly_budget is None else float(monthly_budget)
            )
            st.success(
                f"Confirmed {len(st.session_state[REVIEWED_TRANSACTIONS_KEY])} "
                "reviewed transactions for this session."
            )
        except (KeyError, ValueError) as error:
            st.error(f"Review confirmation failed: {error}")

    if REVIEWED_TRANSACTIONS_KEY in st.session_state:
        st.success(
            "Reviewed transactions are confirmed and stored only in this active "
            "Streamlit session. No report or analytics have been generated yet."
        )
