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
from finpulse.presentation import (  # noqa: E402
    MONTHLY_CATEGORY_COLUMNS,
    OTHER_INCLUDED_DEBITS,
    SPENDING_CATEGORIES,
    build_monthly_spending_view,
    build_overview_metrics,
    format_inr,
    format_percentage,
    largest_spending_category,
    monthly_spending_observations,
    monthly_spending_snapshot,
    statement_period_context,
)
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
    if diagnostics.get("header_row_number"):
        st.caption(
            "Transaction table detected on "
            f"{diagnostics.get('selected_sheet') or 'the uploaded file'}; "
            f"header row {diagnostics['header_row_number']}. "
            f"Ignored {diagnostics.get('metadata_rows_ignored', 0)} metadata rows."
        )

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
            safe_columns = [
                column
                for column in ("_source_row", "_drop_reason")
                if column in rejected.columns
            ]
            st.dataframe(rejected[safe_columns], width="stretch")


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


def _render_report_transactions(report: FinPulseReport) -> None:
    transactions = st.session_state.get(WORKING_TRANSACTIONS_KEY)
    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        transactions = st.session_state.get(REVIEWED_TRANSACTIONS_KEY)
    transactions = initialize_review_categories(transactions)
    if transactions.empty:
        st.info("Reviewed transactions are unavailable for this session.")
        return

    included = int(transactions["include_in_analysis"].sum())
    excluded = len(transactions) - included
    st.subheader("Reviewed Transactions")
    st.caption(
        "Correct a category or exclude exceptional transactions that do not "
        "represent your normal financial behavior."
    )
    st.write(f"**{included} of {len(transactions)} transactions included in analysis**")
    if excluded:
        st.caption(f"{excluded} transactions excluded")

    simple_columns = [
        column
        for column in (
            "date",
            "receiver",
            "final_category",
            "include_in_analysis",
        )
        if column in transactions.columns
    ]
    simple_view = transactions[simple_columns].copy()
    if "amount" in transactions.columns:
        simple_view.insert(2, "amount_display", transactions["amount"].map(format_inr))
    with st.form("report_transaction_review_form"):
        edited = st.data_editor(
            simple_view,
            key=f"report_transaction_editor_{st.session_state.get('review_editor_revision', 0)}",
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            disabled=[
                column
                for column in simple_view.columns
                if column not in {"final_category", "include_in_analysis"}
            ],
            column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "receiver": st.column_config.TextColumn("Receiver"),
                "amount_display": st.column_config.TextColumn("Amount"),
                "final_category": st.column_config.SelectboxColumn(
                    "Category",
                    options=list(FINPULSE_CATEGORIES),
                    required=True,
                ),
                "include_in_analysis": st.column_config.CheckboxColumn(
                    "Include in FinPulse analysis",
                    default=True,
                ),
            },
        )
        save_changes = st.form_submit_button("Save Transaction Changes")

    if save_changes:
        try:
            updated = apply_review_edits(transactions, edited)
            changed = (
                not updated["final_category"].equals(transactions["final_category"])
                or not updated["include_in_analysis"].equals(
                    transactions["include_in_analysis"]
                )
            )
            if changed:
                st.session_state[WORKING_TRANSACTIONS_KEY] = updated
                invalidate_review_confirmation(st.session_state)
                _advance_editor_revision()
                st.session_state["report_transaction_changes_saved"] = True
                st.rerun()
            else:
                st.info("No transaction changes were detected.")
        except (KeyError, TypeError, ValueError) as error:
            st.error(f"Could not save transaction changes: {error}")

    with st.expander("Categorization details"):
        detail_columns = [
            column
            for column in (
                "date",
                "description",
                "detailed_category",
                "predicted_category",
                "confidence",
                "final_category",
                "include_in_analysis",
            )
            if column in transactions.columns
        ]
        st.caption(
            "These fields explain the automatic categorization. Predicted categories "
            "remain unchanged when you correct the final category."
        )
        st.dataframe(
            transactions[detail_columns],
            width="stretch",
            hide_index=True,
        )


def _render_advanced_analysis(report: FinPulseReport) -> None:
    analytics = report.analytics
    statement = analytics.statement_summary
    score = analytics.score_result

    st.subheader("Behavioral Ratios")
    ratio_items = list(report.behavioral_ratios.items())
    for start in range(0, len(ratio_items), 3):
        columns = st.columns(3)
        for column, (label, value) in zip(columns, ratio_items[start : start + 3]):
            column.metric(f"{label} ratio", format_percentage(value))

    st.subheader("Score Components")
    component_names = {
        "spending_control": "Spending Control",
        "saving_investment": "Saving & Investment",
        "repayment_management": "Repayment Management",
    }
    component_columns = st.columns(3)
    for column, (key, component) in zip(
        component_columns, score["components"].items()
    ):
        column.metric(
            component_names[key],
            f"{component['score']} / {component['max_score']}",
        )
        column.caption(component["explanation"])
    st.caption(report.score_presentation.explanation)

    st.subheader("Report Confidence and Coverage")
    quality = analytics.data_quality
    quality_columns = st.columns(3)
    quality_columns[0].metric("Report Confidence", quality["analytical_confidence"])
    quality_columns[1].metric("Included Transactions", statement["transaction_count"])
    quality_columns[2].metric("Coverage", f"{statement['coverage_days']} days")
    for warning in quality["warnings"]:
        st.markdown(f"- {warning}")

    transactions = st.session_state.get(WORKING_TRANSACTIONS_KEY)
    if isinstance(transactions, pd.DataFrame) and "confidence" in transactions:
        st.markdown("#### Categorization Confidence")
        counts = transactions["confidence"].value_counts()
        confidence_columns = st.columns(3)
        for position, confidence in enumerate(("High", "Medium", "Low")):
            confidence_columns[position].metric(
                confidence, int(counts.get(confidence, 0))
            )

    summary = report.review_summary
    st.caption(
        f"Manual corrections: {summary['manually_corrected']} · "
        f"Low-confidence automatic classifications: "
        f"{summary['low_confidence_automatic_classifications']}"
    )

    st.subheader("K-Means Behavioral Segment")
    persona = report.persona
    if persona.persona_available:
        persona_columns = st.columns(2)
        persona_columns[0].metric("Behavioral Segment", persona.persona_name)
        persona_columns[1].metric("Persona Confidence", persona.persona_confidence)
        if persona.persona_description:
            st.write(persona.persona_description)
        st.caption(
            "Prototype label based on synthetic behavioral reference profiles; "
            "not production-validated customer segmentation."
        )
    else:
        st.metric("Behavioral Segment", "Unavailable for uploaded statements")
        for reason in persona.reasons:
            st.markdown(f"- {reason}")
        st.caption(
            "Persona unavailability does not affect the rule-based FinPulse score."
        )

    with st.expander("Technical feature availability"):
        feature_rows = [
            {
                "Feature": name.replace("_", " ").title(),
                "Availability": availability,
            }
            for name, availability in persona.feature_availability.items()
        ]
        st.dataframe(pd.DataFrame(feature_rows), width="stretch", hide_index=True)

    diagnostics = st.session_state.get(DIAGNOSTICS_KEY)
    if diagnostics:
        with st.expander("Statement ingestion diagnostics"):
            st.json(diagnostics)


def _render_full_statement_cash_flow(report: FinPulseReport) -> None:
    with st.expander("Full statement cash flow"):
        cash_flow = report.analytics.statement_cash_flow_summary
        st.caption(statement_period_context(cash_flow))
        cash_columns = st.columns(3)
        cash_columns[0].metric(
            "Observed Credits", format_inr(cash_flow["observed_credits"])
        )
        cash_columns[1].metric(
            "Observed Debits", format_inr(cash_flow["observed_debits"])
        )
        cash_columns[2].metric(
            "Net Statement Cash Flow",
            format_inr(cash_flow["net_statement_cash_flow"]),
        )
        st.caption(
            "Statement cash flow describes what happened in the full uploaded "
            "statement. Monthly behavioral spending uses included debit "
            "transactions only."
        )


def _monthly_chart_categories(monthly_view: pd.DataFrame) -> list[str]:
    categories = list(MONTHLY_CATEGORY_COLUMNS)
    if float(monthly_view[OTHER_INCLUDED_DEBITS].sum()) > 0:
        categories.append(OTHER_INCLUDED_DEBITS)
    return categories


def _render_one_month_spending(
    report: FinPulseReport,
    monthly_view: pd.DataFrame,
) -> None:
    snapshot = monthly_spending_snapshot(
        monthly_view, str(monthly_view.iloc[0]["month_key"])
    )
    st.subheader("Monthly Spending")
    st.caption(
        f"Included debit spending observed in {snapshot['Month']}."
    )
    top_columns = st.columns(2)
    top_columns[0].metric(
        "Monthly Spending", format_inr(snapshot["Monthly Spending"])
    )
    top_columns[1].metric("Included Transactions", snapshot["Transactions"])
    if snapshot["Partial Month"]:
        st.caption(
            f"Partial month: {snapshot['Coverage Days']} of "
            f"{snapshot['Days in Month']} days represented. Values are observed "
            "amounts and are not extrapolated."
        )

    categories = _monthly_chart_categories(monthly_view)
    category_rows = [
        {"Category": category, "Amount": float(snapshot[category])}
        for category in categories
    ]
    category_frame = pd.DataFrame(category_rows)
    display_frame = category_frame.copy()
    display_frame["Amount"] = display_frame["Amount"].map(format_inr)
    st.dataframe(display_frame, width="stretch", hide_index=True)

    positive = category_frame.loc[category_frame["Amount"] > 0]
    if not positive.empty:
        figure = px.pie(positive, names="Category", values="Amount", hole=0.55)
        figure.update_layout(
            legend_title_text="",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(figure, width="stretch")

    if snapshot["Budget Used"] is not None:
        st.metric("Budget Used", format_percentage(snapshot["Budget Used"]))
    _render_full_statement_cash_flow(report)


def _render_multi_month_spending(
    report: FinPulseReport,
    monthly_view: pd.DataFrame,
) -> None:
    st.subheader("Your Spending Over Time")
    st.caption(
        f"Based on {len(monthly_view)} months of included debit transactions."
    )

    trend = px.line(
        monthly_view,
        x="Axis Month",
        y="Monthly Spending",
        markers=True,
        custom_data=["Month Selector", "Coverage Days", "Days in Month"],
    )
    trend.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Monthly spending: ₹%{y:,.0f}"
            "<br>Coverage: %{customdata[1]} of %{customdata[2]} days<extra></extra>"
        )
    )
    budget = report.analytics.budget_summary
    if budget is not None:
        trend.add_hline(
            y=budget["monthly_budget"],
            line_dash="dash",
            annotation_text="Monthly budget",
        )
    trend.update_layout(
        xaxis_title="",
        yaxis_title="Included debit spending (₹)",
        margin=dict(l=10, r=10, t=25, b=10),
        hovermode="x unified",
    )
    trend.update_yaxes(tickprefix="₹", separatethousands=True)
    st.plotly_chart(trend, width="stretch")
    if bool(monthly_view["Partial Month"].any()):
        st.caption(
            "* Partial month. The chart shows actual included spending observed "
            "during the covered days; no full-month extrapolation is applied."
        )

    st.markdown("#### Spending Categories Over Time")
    categories = _monthly_chart_categories(monthly_view)
    category_long = monthly_view[["Axis Month", *categories]].melt(
        id_vars="Axis Month",
        var_name="Category",
        value_name="Amount",
    )
    category_chart = px.bar(
        category_long,
        x="Axis Month",
        y="Amount",
        color="Category",
        barmode="stack",
    )
    category_chart.update_layout(
        xaxis_title="",
        yaxis_title="Included debit spending (₹)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    category_chart.update_yaxes(tickprefix="₹", separatethousands=True)
    st.plotly_chart(category_chart, width="stretch")

    st.markdown("#### View a Month")
    selection = st.selectbox(
        "View a month",
        options=monthly_view["Month Selector"].tolist(),
        label_visibility="collapsed",
        key="monthly_spending_month_selector",
    )
    snapshot = monthly_spending_snapshot(monthly_view, selection)
    snapshot_values = [
        ("Monthly Spending", snapshot["Monthly Spending"]),
        *[(category, snapshot[category]) for category in SPENDING_CATEGORIES],
    ]
    if snapshot[OTHER_INCLUDED_DEBITS] > 0:
        snapshot_values.append(
            (OTHER_INCLUDED_DEBITS, snapshot[OTHER_INCLUDED_DEBITS])
        )
    for start in range(0, len(snapshot_values), 3):
        columns = st.columns(3)
        for column, (label, value) in zip(columns, snapshot_values[start : start + 3]):
            column.metric(label, format_inr(value))
    if snapshot["Budget Used"] is not None:
        st.metric("Budget Used", format_percentage(snapshot["Budget Used"]))
    if snapshot["Partial Month"]:
        st.caption(
            f"Partial month: {snapshot['Coverage Days']} of "
            f"{snapshot['Days in Month']} days represented. Snapshot values are "
            "actual observed amounts, not extrapolated estimates."
        )
    else:
        st.caption(
            f"{snapshot['Transactions']} included debit transactions in "
            f"{snapshot['Month']}."
        )

    composition_rows = [
        {"Category": category, "Amount": float(snapshot[category])}
        for category in categories
    ]
    composition = pd.DataFrame(composition_rows)
    composition = composition.loc[composition["Amount"] > 0]
    if not composition.empty:
        composition_chart = px.pie(
            composition,
            names="Category",
            values="Amount",
            hole=0.55,
        )
        composition_chart.update_layout(
            legend_title_text="",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(composition_chart, width="stretch")

    observations = monthly_spending_observations(monthly_view)
    if observations:
        st.markdown("#### Your Monthly Story")
        for observation in observations:
            st.markdown(f"- {observation}")
    _render_full_statement_cash_flow(report)


def _render_report(report: FinPulseReport) -> None:
    analytics = report.analytics
    statement = analytics.statement_summary
    overview = build_overview_metrics(statement, analytics.score_result)

    st.divider()
    st.header("Your FinPulse Dashboard")
    st.caption(
        "Based on your confirmed categories and transaction inclusion choices."
    )
    overview_tab, spending_tab, improve_tab, transactions_tab, advanced_tab = st.tabs(
        (
            "Overview",
            "My Spending",
            "Improve",
            "Transactions",
            "Advanced Analysis",
        )
    )

    with overview_tab:
        first_row = st.columns(2)
        first_row[0].metric(
            "Monthly Available Amount",
            format_inr(overview["monthly_available_amount"]),
        )
        first_row[1].metric(
            overview["spending_label"], format_inr(overview["monthly_spending"])
        )
        first_row[1].caption(overview["period_context"])

        second_row = st.columns(2)
        amount_left = overview["estimated_amount_left"]
        second_row[0].metric("Estimated Amount Left", format_inr(amount_left))
        if amount_left is not None and float(amount_left) < 0:
            second_row[0].caption(
                f"Overspending by {format_inr(abs(float(amount_left)))} per month."
            )
        else:
            second_row[0].caption("After average monthly included debit spending.")
        score_text = (
            f"{overview['score']} / 100"
            if overview["score"] is not None
            else "Unavailable"
        )
        second_row[1].metric("FinPulse Score", score_text)
        second_row[1].caption(str(overview["score_band"] or ""))

        budget = analytics.budget_summary
        if budget is not None:
            st.markdown("#### Monthly Spending Budget")
            budget_columns = st.columns(2)
            budget_columns[0].metric(
                "Monthly Spending Budget", format_inr(budget["monthly_budget"])
            )
            budget_columns[1].metric(
                "Budget Used",
                format_percentage(
                    budget["budget_utilization_percent"], already_percent=True
                ),
            )
            status = "over" if budget["over_budget"] else "within"
            st.caption(
                f"Average monthly spending is {status} the supplied budget by "
                f"{format_inr(abs(budget['budget_variance']))}."
            )

        st.markdown("#### Where Your Money Goes")
        spending = report.debit_spending_breakdown.copy()
        display_categories = list(SPENDING_CATEGORIES)
        if analytics.category_summary["Income"]["monthly_normalized"] > 0:
            display_categories.append("Income")
        spending = spending.loc[
            spending["category"].isin(display_categories)
            & (spending["monthly_normalized"] > 0)
        ]
        spending.loc[
            spending["category"] == "Income", "category"
        ] = "Income-labelled debit"
        if spending.empty:
            st.info("No included debit spending is available to chart.")
        else:
            figure = px.pie(
                spending,
                names="category",
                values="monthly_normalized",
                hole=0.55,
            )
            figure.update_layout(
                legend_title_text="",
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(figure, width="stretch")
            largest = largest_spending_category(analytics.category_summary)
            if largest:
                st.caption(
                    f"Most of your average monthly spending goes toward {largest[0]}."
                )

    with spending_tab:
        monthly_budget = (
            analytics.budget_summary["monthly_budget"]
            if analytics.budget_summary is not None
            else None
        )
        monthly_view = build_monthly_spending_view(
            analytics.monthly_summary,
            monthly_budget,
        )
        if statement["transaction_month_count"] > 1:
            _render_multi_month_spending(report, monthly_view)
        else:
            _render_one_month_spending(report, monthly_view)

    with improve_tab:
        st.subheader("What to Focus On")
        if analytics.signals:
            for signal in analytics.signals:
                st.markdown(f"- {signal}")
        else:
            st.info("No additional supported behavioral signals were identified.")

        st.subheader("Recommendations")
        if analytics.recommendations:
            for recommendation in analytics.recommendations:
                st.markdown(f"**{recommendation['category']}**")
                st.write(recommendation["recommendation"])
        else:
            st.info("No additional deterministic recommendations were generated.")
        st.caption(
            "FinPulse provides behavioral analytics for informational and educational "
            "purposes and is not financial advice."
        )

    with transactions_tab:
        _render_report_transactions(report)

    with advanced_tab:
        _render_advanced_analysis(report)


st.set_page_config(
    page_title="Analyze Statement | FinPulse",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Analyze Your Bank Statement")
st.caption(
    "Your uploaded statement is processed for this session and is not written "
    "to the FinPulse database. CSV, XLS, and XLSX are supported; PDF is not yet supported."
)

existing_report = st.session_state.get(REPORT_KEY)
if isinstance(existing_report, FinPulseReport):
    if st.button("Update Statement or Inputs"):
        invalidate_report_state(st.session_state)
        st.rerun()
    _render_report(existing_report)
    st.stop()

if st.session_state.pop("report_transaction_changes_saved", False):
    st.info(
        "Transaction changes were saved. Confirm the reviewed transactions and "
        "generate the report again to refresh the dashboard."
    )

with st.form("statement_setup_form"):
    uploaded_file = st.file_uploader(
        "Upload Statement",
        type=["csv", "xls", "xlsx"],
        help="CSV, XLS, and XLSX statements are supported. PDF support will be added later.",
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
        st.error("Upload a CSV, XLS, or XLSX bank statement before analyzing.")
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
        "Correct a category or exclude exceptional transactions that do not "
        "represent your normal financial behavior."
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
            "final_category",
            "include_in_analysis",
        ]
        display_columns = [
            column for column in display_columns if column in review_view.columns
        ]
        review_display = review_view[display_columns].copy()
        if "amount" in review_view.columns:
            review_display.insert(
                2, "amount_display", review_view["amount"].map(format_inr)
            )
        editor_key = (
            f"statement_review_editor_{review_mode}_"
            f"{st.session_state['review_editor_revision']}"
        )
        edited_view = st.data_editor(
            review_display,
            key=editor_key,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            disabled=[
                column
                for column in review_display.columns
                if column not in {"final_category", "include_in_analysis"}
            ],
            column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "receiver": st.column_config.TextColumn("Receiver"),
                "amount_display": st.column_config.TextColumn("Amount"),
                "final_category": st.column_config.SelectboxColumn(
                    "Category",
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

        with st.expander("Categorization details"):
            technical_columns = [
                column
                for column in (
                    "date",
                    "description",
                    "transaction_type",
                    "detailed_category",
                    "predicted_category",
                    "confidence",
                    "final_category",
                )
                if column in review_view.columns
            ]
            st.caption(
                "Automatic categorization details are shown for transparency. "
                "Your reviewed Category is the value used by FinPulse."
            )
            st.dataframe(
                review_view[technical_columns],
                width="stretch",
                hide_index=True,
            )

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
