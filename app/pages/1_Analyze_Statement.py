"""Session-only bank statement analysis and personalized FinPulse report."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finpulse.categorization import categorize_transactions  # noqa: E402
from finpulse.reporting import (  # noqa: E402
    ANALYTICS_KEY,
    FILE_FINGERPRINT_KEY,
    MONTHLY_AMOUNT_KEY,
    MONTHLY_BUDGET_KEY,
    PERSONA_KEY,
    REPORT_KEY,
    REPORT_SIGNATURE_KEY,
    FinPulseReport,
    ReportNotReadyError,
    build_report_signature,
    generate_financial_report,
    invalidate_report_state,
    invalidate_review_confirmation,
    synchronize_statement_inputs,
)
from finpulse.review import (  # noqa: E402
    FINPULSE_CATEGORIES,
    apply_review_edits,
    filter_review_transactions,
    initialize_review_categories,
    validate_review_fields,
)
from finpulse.statement_ingestion import (  # noqa: E402
    IngestionResult,
    StatementReadError,
    build_manual_column_mapping,
    read_statement,
)


WORKING_TRANSACTIONS_KEY = "categorized_transactions"
REVIEWED_TRANSACTIONS_KEY = "reviewed_transactions"
DIAGNOSTICS_KEY = "statement_diagnostics"
UPLOAD_BYTES_KEY = "statement_upload_bytes"
SOURCE_NAME_KEY = "statement_source_name"
REJECTED_ROWS_KEY = "statement_rejected_rows"
MANUAL_MAPPING_KEY = "manual_mapping_required"


def _advance_editor_revision() -> None:
    st.session_state["review_editor_revision"] = (
        st.session_state.get("review_editor_revision", 0) + 1
    )


def _file_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _accept_ingestion(result: IngestionResult) -> None:
    st.session_state[DIAGNOSTICS_KEY] = asdict(result.diagnostics)
    st.session_state[REJECTED_ROWS_KEY] = result.rejected_rows.copy()
    amount_layout = result.mapping_result.amount_layout
    required_mapping_fields = {"date", "description"}
    if amount_layout == "separate_debit_credit":
        required_mapping_fields.update(("debit", "credit"))
    elif amount_layout == "amount_and_transaction_type":
        required_mapping_fields.update(("amount", "transaction_type"))
    else:
        required_mapping_fields.update(
            ("debit", "credit", "amount", "transaction_type")
        )
    required_ambiguities = required_mapping_fields.intersection(
        result.mapping_result.ambiguous_mappings
    )
    mapping_needs_review = (
        not result.mapping_result.is_valid or bool(required_ambiguities)
    )
    if mapping_needs_review:
        st.session_state[MANUAL_MAPPING_KEY] = True
        return
    if result.transactions.empty:
        st.session_state[MANUAL_MAPPING_KEY] = False
        return

    categorized = categorize_transactions(result.transactions)
    st.session_state[WORKING_TRANSACTIONS_KEY] = initialize_review_categories(
        categorized
    )
    st.session_state.pop(REVIEWED_TRANSACTIONS_KEY, None)
    st.session_state[MANUAL_MAPPING_KEY] = False
    invalidate_report_state(st.session_state)
    _advance_editor_revision()


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
            "Some source columns are ambiguous. Select the intended columns "
            "in Manual Column Mapping below."
        )

    with st.expander("Detected mapping and validation details"):
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

    rejected = st.session_state.get(REJECTED_ROWS_KEY)
    if isinstance(rejected, pd.DataFrame) and not rejected.empty:
        with st.expander("Inspect rows excluded during cleaning"):
            st.dataframe(rejected, width="stretch")


def _mapping_select(
    label: str,
    field: str,
    columns: list[str],
    diagnostics: dict[str, object],
    *,
    optional: bool = False,
) -> str | None:
    options = (["Not mapped"] if optional else []) + columns
    detected = diagnostics["detected_column_mapping"].get(field)
    candidates = diagnostics["candidate_columns"].get(field, ())
    preferred = detected or (candidates[0] if candidates else None)
    default_index = options.index(preferred) if preferred in options else 0
    selected = st.selectbox(
        label,
        options,
        index=default_index,
        key=f"manual_mapping_{field}",
    )
    return None if selected == "Not mapped" else selected


def _render_manual_mapping(diagnostics: dict[str, object]) -> None:
    data = st.session_state.get(UPLOAD_BYTES_KEY)
    source_name = st.session_state.get(SOURCE_NAME_KEY)
    columns = list(diagnostics.get("source_columns", ()))
    if not isinstance(data, bytes) or not source_name or not columns:
        st.error("Manual mapping is unavailable. Upload the statement again.")
        return

    st.divider()
    st.subheader("Manual Column Mapping")
    st.caption(
        "Choose which source columns represent the required FinPulse fields. "
        "This reruns the same standard ingestion and cleaning pipeline."
    )
    with st.form("manual_column_mapping_form"):
        first, second = st.columns(2)
        with first:
            date_column = _mapping_select("Date", "date", columns, diagnostics)
        with second:
            description_column = _mapping_select(
                "Description / remarks", "description", columns, diagnostics
            )

        separate_available = bool(
            diagnostics["candidate_columns"].get("debit")
            or diagnostics["candidate_columns"].get("credit")
        )
        amount_layout = st.radio(
            "Amount layout",
            ("Separate debit and credit columns", "Amount and transaction type"),
            index=0 if separate_available else 1,
            horizontal=True,
        )

        debit_column = credit_column = amount_column = type_column = None
        if amount_layout == "Separate debit and credit columns":
            first, second = st.columns(2)
            with first:
                debit_column = _mapping_select(
                    "Debit / withdrawal",
                    "debit",
                    columns,
                    diagnostics,
                    optional=True,
                )
            with second:
                credit_column = _mapping_select(
                    "Credit / deposit",
                    "credit",
                    columns,
                    diagnostics,
                    optional=True,
                )
        else:
            first, second = st.columns(2)
            with first:
                amount_column = _mapping_select(
                    "Transaction amount", "amount", columns, diagnostics
                )
            with second:
                type_column = _mapping_select(
                    "Debit / credit indicator",
                    "transaction_type",
                    columns,
                    diagnostics,
                )

        first, second = st.columns(2)
        with first:
            transaction_id_column = _mapping_select(
                "Transaction ID (optional)",
                "transaction_id",
                columns,
                diagnostics,
                optional=True,
            )
        with second:
            balance_column = _mapping_select(
                "Balance (optional)",
                "balance",
                columns,
                diagnostics,
                optional=True,
            )
        apply_mapping = st.form_submit_button(
            "Apply Mapping and Continue", type="primary"
        )

    if apply_mapping:
        try:
            mapping = build_manual_column_mapping(
                columns,
                date=date_column,
                description=description_column,
                debit=debit_column,
                credit=credit_column,
                amount=amount_column,
                transaction_type=type_column,
                transaction_id=transaction_id_column,
                balance=balance_column,
            )
            result = read_statement(data, filename=source_name, mapping=mapping)
            _accept_ingestion(result)
            if result.is_usable:
                st.success("Column mapping accepted. Transactions are ready for review.")
            else:
                st.error(
                    "The selected mapping produced no usable transactions. "
                    "Review the updated diagnostics and try again."
                )
        except (StatementReadError, TypeError, ValueError) as error:
            st.error(f"Could not apply the selected mapping: {error}")


def _statement_coverage(transactions: pd.DataFrame) -> tuple[int, str, int]:
    transaction_count = len(transactions)
    parsed_dates = pd.to_datetime(transactions["date"], errors="coerce").dropna()
    if parsed_dates.empty:
        return transaction_count, "Unavailable", 0
    start = parsed_dates.min().date()
    end = parsed_dates.max().date()
    return (
        transaction_count,
        f"{start.isoformat()} to {end.isoformat()}",
        (end - start).days + 1,
    )


def _money(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"₹{float(value):,.2f}"


def _percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value) * 100:.1f}%"


def _render_report(report: FinPulseReport) -> None:
    analytics = report.analytics
    statement = analytics.statement_summary

    st.divider()
    st.header("Your FinPulse Report")
    st.caption(
        "Generated from your confirmed categories and inclusion choices for this session."
    )

    st.subheader("1. Financial Snapshot")
    snapshot = st.columns(5)
    snapshot[0].metric(
        "Monthly Available Amount", _money(statement.get("monthly_available_amount"))
    )
    snapshot[1].metric(
        "Monthly-normalized spending",
        _money(statement.get("monthly_normalized_debit_spending")),
    )
    snapshot[2].metric(
        "Estimated remaining", _money(statement.get("remaining_amount_estimate"))
    )
    snapshot[3].metric(
        "Analyzed debit coverage", f"{statement['coverage_days']} days"
    )
    snapshot[4].metric("Analyzed transactions", statement["transaction_count"])

    budget = analytics.budget_summary
    if budget is not None:
        budget_columns = st.columns(4)
        budget_columns[0].metric("Monthly Budget", _money(budget["monthly_budget"]))
        budget_columns[1].metric(
            "Budget Utilization", f"{budget['budget_utilization_percent']:.1f}%"
        )
        budget_columns[2].metric(
            "Budget Variance", _money(abs(budget["budget_variance"]))
        )
        budget_columns[3].metric(
            "Budget Status", "Over budget" if budget["over_budget"] else "Within budget"
        )
        variance_direction = "over" if budget["over_budget"] else "under"
        st.caption(
            f"Monthly-normalized spending is {_money(abs(budget['budget_variance']))} "
            f"{variance_direction} the supplied budget."
        )

    st.markdown("#### Statement Cash Flow (informational)")
    cash_flow = analytics.statement_cash_flow_summary
    cash_columns = st.columns(4)
    cash_columns[0].metric("Observed credits", _money(cash_flow["observed_credits"]))
    cash_columns[1].metric("Observed debits", _money(cash_flow["observed_debits"]))
    cash_columns[2].metric(
        "Net statement cash flow", _money(cash_flow["net_statement_cash_flow"])
    )
    cash_columns[3].metric(
        "Spending considered for analysis",
        _money(cash_flow["spending_considered_for_analysis"]),
    )
    st.caption(
        "Statement cash flow describes what happened in the account and includes "
        "the full reviewed statement. FinPulse behavioral analytics use the "
        "user-declared Monthly Available Amount and included debit transactions only."
    )

    st.subheader("2. Spending Breakdown")
    spending = report.debit_spending_breakdown.copy()
    positive_spending = spending.loc[spending["monthly_normalized"] > 0]
    if positive_spending.empty:
        st.info("No categorized debit spending is available to chart.")
    else:
        figure = px.bar(
            positive_spending,
            x="category",
            y="monthly_normalized",
            color="category",
            labels={
                "category": "Category",
                "monthly_normalized": "Monthly-normalized amount",
            },
        )
        figure.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        figure.update_yaxes(tickprefix="₹", separatethousands=True)
        st.plotly_chart(figure, width="stretch")
    zero_categories = spending.loc[
        spending["monthly_normalized"] == 0, "category"
    ].tolist()
    if zero_categories:
        st.caption("No observed debit spending: " + ", ".join(zero_categories))
    display_breakdown = spending[
        ["category", "monthly_normalized", "total", "transaction_count"]
    ].rename(
        columns={
            "category": "Category",
            "monthly_normalized": "Monthly normalized",
            "total": "Statement total",
            "transaction_count": "Transactions",
        }
    )
    st.dataframe(
        display_breakdown,
        width="stretch",
        hide_index=True,
        column_config={
            "Monthly normalized": st.column_config.NumberColumn(format="₹ %.2f"),
            "Statement total": st.column_config.NumberColumn(format="₹ %.2f"),
        },
    )

    st.subheader("3. Behavioral Ratios")
    ratio_items = list(report.behavioral_ratios.items())
    for start in range(0, len(ratio_items), 3):
        columns = st.columns(3)
        for column, (label, value) in zip(columns, ratio_items[start : start + 3]):
            column.metric(f"{label} ratio", _percent(value))

    st.subheader("4. FinPulse Behavioral Score")
    score = analytics.score_result
    score_columns = st.columns([2, 2, 3])
    score_columns[0].metric(
        "FinPulse Behavioral Score", f"{score['finpulse_score']} / 100"
    )
    score_columns[1].metric("Score Band", score["score_band"])
    score_columns[2].metric("Score Status", report.score_presentation.status_label)
    if report.score_presentation.is_provisional:
        st.warning(report.score_presentation.explanation)
    else:
        st.caption(report.score_presentation.explanation)

    component_names = {
        "spending_control": "Spending Control",
        "savings": "Savings",
        "debt_management": "Debt Management",
        "stability": "Stability",
    }
    component_columns = st.columns(4)
    for column, (key, component) in zip(
        component_columns, score["components"].items()
    ):
        if component["available"]:
            value = f"{component['score']} / {component['maximum']}"
            help_text = None
        else:
            value = "Unavailable"
            help_text = component.get("unavailable_reason")
        column.metric(component_names[key], value, help=help_text)

    st.subheader("5. Data / Report Confidence")
    quality = analytics.data_quality
    st.metric("Behavioral report confidence", quality["analytical_confidence"])
    st.caption(
        "This describes statement coverage and behavioral evidence. It is separate "
        "from transaction categorization confidence and persona confidence."
    )
    for warning in quality["warnings"]:
        st.markdown(f"- {warning}")

    st.subheader("6. Behavioral Signals")
    if analytics.signals:
        for signal in analytics.signals:
            st.markdown(f"- {signal}")
    else:
        st.info("No additional supported behavioral signals were identified.")

    st.subheader("7. Recommendations")
    if analytics.recommendations:
        for recommendation in analytics.recommendations:
            st.markdown(f"**{recommendation['category']}**")
            st.write(recommendation["recommendation"])
    else:
        st.info("No additional deterministic recommendations were generated.")
    st.caption(
        "FinPulse provides behavioral analytics for informational purposes and is "
        "not financial advice."
    )

    st.subheader("8. K-Means Behavioral Segment")
    persona = report.persona
    if persona.persona_available:
        st.caption("Prototype ML segment")
        persona_columns = st.columns([3, 1])
        persona_columns[0].metric("Behavioral Segment", persona.persona_name)
        persona_columns[1].metric("Persona Confidence", persona.persona_confidence)
        if persona.persona_description:
            st.write(persona.persona_description)
        st.caption(
            "Based on similarity to synthetic behavioral reference profiles. "
            "This is a supporting prototype label, not a scientifically validated "
            "consumer type."
        )
        for reason in persona.reasons:
            st.caption(reason)
    else:
        credit_incompatibility = any(
            "bank credits" in reason.casefold() for reason in persona.reasons
        )
        unavailable_label = (
            "Unavailable for uploaded statements"
            if credit_incompatibility
            else "Needs more history"
        )
        st.metric("Behavioral Segment", unavailable_label)
        for reason in persona.reasons:
            st.markdown(f"- {reason}")
        st.caption(
            "Persona unavailability does not affect the rule-based FinPulse score."
        )

    st.subheader("9. Transaction Review Summary")
    summary = report.review_summary
    summary_metrics = [
        ("Total transactions", summary["transaction_count"]),
        ("Included in analysis", summary["included_in_analysis"]),
    ]
    if summary["excluded_from_analysis"]:
        summary_metrics.append(
            ("Excluded from analysis", summary["excluded_from_analysis"])
        )
    summary_metrics.extend(
        (
            ("Manual corrections", summary["manually_corrected"]),
            (
                "Low-confidence classifications",
                summary["low_confidence_automatic_classifications"],
            ),
        )
    )
    review_columns = st.columns(len(summary_metrics))
    for column, (label, value) in zip(review_columns, summary_metrics):
        column.metric(label, value)
    if summary["excluded_from_analysis"]:
        with st.expander("View transactions excluded from analysis"):
            excluded_columns = [
                column
                for column in (
                    "date",
                    "receiver",
                    "description",
                    "amount",
                    "transaction_type",
                    "final_category",
                    "confidence",
                    "include_in_analysis",
                )
                if column in report.excluded_transactions.columns
            ]
            st.dataframe(
                report.excluded_transactions[excluded_columns],
                width="stretch",
                hide_index=True,
            )


st.set_page_config(
    page_title="Analyze Statement | FinPulse",
    page_icon="📄",
    layout="wide",
)

st.title("Analyze Your Bank Statement")
st.caption(
    "Your uploaded statement is processed for this session and is not written "
    "to the FinPulse database. CSV and XLSX are supported; PDF is not yet supported."
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
    analyze_statement = st.form_submit_button("Analyze Statement", type="primary")


if analyze_statement:
    if uploaded_file is None:
        st.error("Upload a CSV or XLSX bank statement before analyzing.")
    elif monthly_available_amount is None or monthly_available_amount <= 0:
        st.error("Monthly Available Amount is required and must be greater than zero.")
    elif monthly_budget is not None and monthly_budget <= 0:
        st.error("Monthly Spending Budget must be greater than zero or left blank.")
    else:
        statement_bytes = uploaded_file.getvalue()
        fingerprint = _file_fingerprint(statement_bytes)
        changes = synchronize_statement_inputs(
            st.session_state,
            file_fingerprint=fingerprint,
            monthly_available_amount=float(monthly_available_amount),
            monthly_budget=None if monthly_budget is None else float(monthly_budget),
        )
        st.session_state[SOURCE_NAME_KEY] = uploaded_file.name
        st.session_state[UPLOAD_BYTES_KEY] = statement_bytes

        already_ingested = (
            "uploaded_file" not in changes
            and isinstance(st.session_state.get(WORKING_TRANSACTIONS_KEY), pd.DataFrame)
        )
        if already_ingested:
            if changes:
                st.info(
                    "Your reviewed categories were preserved. Previous report "
                    "analytics were cleared because report inputs changed."
                )
            else:
                st.info("This statement is already ready for review.")
        else:
            try:
                ingestion_result = read_statement(
                    statement_bytes, filename=uploaded_file.name
                )
                _accept_ingestion(ingestion_result)
                if st.session_state.get(MANUAL_MAPPING_KEY):
                    st.error(
                        "The statement could not be standardized automatically. "
                        "Use Manual Column Mapping below."
                    )
                elif ingestion_result.transactions.empty:
                    st.error(
                        "No usable transactions remained after cleaning. Review "
                        "the ingestion diagnostics and source statement."
                    )
            except (StatementReadError, TypeError, ValueError) as error:
                st.error(f"Could not analyze this statement: {error}")


diagnostics = st.session_state.get(DIAGNOSTICS_KEY)
if diagnostics:
    _render_diagnostics(diagnostics)
    if st.session_state.get(MANUAL_MAPPING_KEY):
        _render_manual_mapping(diagnostics)


working_transactions = st.session_state.get(WORKING_TRANSACTIONS_KEY)
if isinstance(working_transactions, pd.DataFrame) and not working_transactions.empty:
    working_transactions = initialize_review_categories(working_transactions)
    st.session_state[WORKING_TRANSACTIONS_KEY] = working_transactions
    st.divider()
    st.subheader("Statement Coverage")
    transaction_count, date_range_text, coverage_days = _statement_coverage(
        working_transactions
    )
    coverage_columns = st.columns(3)
    coverage_columns[0].metric("Transactions", transaction_count)
    coverage_columns[1].metric("Approximate coverage", f"{coverage_days} days")
    coverage_columns[2].metric(
        "Low-confidence rows", int((working_transactions["confidence"] == "Low").sum())
    )
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
            + "). You can continue, but the behavioral report will have lower "
            "confidence."
        )
    else:
        st.success(
            "The statement meets the recommended 30-day and 30-transaction coverage."
        )

    st.divider()
    st.subheader("Categorization Summary")
    confidence_counts = working_transactions["confidence"].value_counts()
    confidence_columns = st.columns(3)
    for position, confidence in enumerate(("High", "Medium", "Low")):
        confidence_columns[position].metric(
            f"{confidence} confidence", int(confidence_counts.get(confidence, 0))
        )

    st.divider()
    st.subheader("Review Transactions")
    st.caption(
        "Predicted categories remain unchanged. Edit Final Category where a "
        "correction is needed, then explicitly confirm the reviewed data."
    )
    st.info(
        "Exclude exceptional transactions that do not represent your normal "
        "financial behavior, such as payments made on someone else's behalf or "
        "temporary money transfers. Exclusions should reflect context, not an "
        "attempt to improve the score."
    )
    st.session_state.setdefault("review_editor_revision", 0)
    review_mode = st.radio(
        "Rows to show",
        options=("All transactions", "Needs review (Low confidence)"),
        horizontal=True,
        key="review_filter_mode",
        on_change=_advance_editor_revision,
    )
    review_view = filter_review_transactions(
        working_transactions,
        low_confidence_only=review_mode == "Needs review (Low confidence)",
    )
    if review_mode == "Needs review (Low confidence)" and review_view.empty:
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
            "include_in_analysis",
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
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            disabled=[
                column
                for column in display_columns
                if column not in {"final_category", "include_in_analysis"}
            ],
            column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "amount": st.column_config.NumberColumn("Amount", format="₹ %.2f"),
                "final_category": st.column_config.SelectboxColumn(
                    "Final Category",
                    options=list(FINPULSE_CATEGORIES),
                    required=True,
                    help="This reviewed category is used by all report analytics.",
                ),
                "include_in_analysis": st.column_config.CheckboxColumn(
                    "Include in FinPulse analysis",
                    help=(
                        "Turn off only for exceptional transactions that do not "
                        "represent your own financial behavior."
                    ),
                    default=True,
                ),
            },
        )
        try:
            updated_transactions = apply_review_edits(
                working_transactions, edited_view
            )
            review_changed = (
                not updated_transactions["final_category"].equals(
                    working_transactions["final_category"]
                )
                or not updated_transactions["include_in_analysis"].equals(
                    working_transactions["include_in_analysis"]
                )
            )
            st.session_state[WORKING_TRANSACTIONS_KEY] = updated_transactions
            working_transactions = updated_transactions
            if review_changed:
                invalidate_review_confirmation(st.session_state)
        except (KeyError, TypeError, ValueError) as error:
            st.error(f"Could not apply review edits: {error}")

    included_review_count = int(working_transactions["include_in_analysis"].sum())
    excluded_review_count = len(working_transactions) - included_review_count
    if excluded_review_count:
        st.caption(
            f"Current review: {included_review_count} included and "
            f"{excluded_review_count} excluded from behavioral analysis."
        )
    if included_review_count == 0:
        st.error(
            "At least one transaction must be included before a FinPulse report "
            "can be generated."
        )

    st.markdown("### Confirm Reviewed Transactions")
    st.caption(
        "The report remains hidden until you explicitly confirm the final categories "
        "and inclusion choices."
    )
    if st.button("Confirm Reviewed Transactions", type="primary"):
        try:
            validate_review_fields(st.session_state[WORKING_TRANSACTIONS_KEY])
            invalidate_report_state(st.session_state)
            st.session_state[REVIEWED_TRANSACTIONS_KEY] = st.session_state[
                WORKING_TRANSACTIONS_KEY
            ].copy(deep=True)
            st.success(
                f"Confirmed {len(st.session_state[REVIEWED_TRANSACTIONS_KEY])} "
                "reviewed transactions for this session."
            )
        except (KeyError, ValueError) as error:
            st.error(f"Review confirmation failed: {error}")


reviewed_transactions = st.session_state.get(REVIEWED_TRANSACTIONS_KEY)
if isinstance(reviewed_transactions, pd.DataFrame) and not reviewed_transactions.empty:
    reviewed_transactions = initialize_review_categories(reviewed_transactions)
    st.session_state[REVIEWED_TRANSACTIONS_KEY] = reviewed_transactions
    try:
        current_signature = build_report_signature(
            file_fingerprint=st.session_state[FILE_FINGERPRINT_KEY],
            reviewed_transactions=reviewed_transactions,
            monthly_available_amount=st.session_state[MONTHLY_AMOUNT_KEY],
            monthly_budget=st.session_state.get(MONTHLY_BUDGET_KEY),
        )
        stored_signature = st.session_state.get(REPORT_SIGNATURE_KEY)
        if stored_signature is not None and stored_signature != current_signature:
            invalidate_report_state(st.session_state)
    except (KeyError, TypeError, ValueError):
        invalidate_report_state(st.session_state)
        current_signature = None

    st.divider()
    st.subheader("Generate FinPulse Report")
    st.caption(
        "The rule-based score is the primary explainable result. K-Means persona "
        "inference remains unavailable for uploaded statements because bank credits "
        "are not used to derive the required income-variability feature."
    )
    if st.button("Generate FinPulse Report", type="primary"):
        try:
            report = generate_financial_report(
                reviewed_transactions,
                st.session_state[MONTHLY_AMOUNT_KEY],
                st.session_state.get(MONTHLY_BUDGET_KEY),
                review_confirmed=True,
            )
            st.session_state[REPORT_KEY] = report
            st.session_state[ANALYTICS_KEY] = report.analytics
            st.session_state[PERSONA_KEY] = report.persona
            st.session_state[REPORT_SIGNATURE_KEY] = current_signature
        except (ReportNotReadyError, FileNotFoundError, TypeError, ValueError) as error:
            st.error(f"The report could not be generated: {error}")
        except Exception:
            st.error(
                "The report could not be generated from this statement. "
                "Please review the inputs and try again."
            )


generated_report = st.session_state.get(REPORT_KEY)
if isinstance(generated_report, FinPulseReport):
    _render_report(generated_report)
