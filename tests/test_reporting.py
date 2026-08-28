from io import BytesIO
import sqlite3
import unittest
from unittest import mock

import pandas as pd

from finpulse.reporting import (
    ANALYTICS_KEY,
    FILE_FINGERPRINT_KEY,
    MONTHLY_AMOUNT_KEY,
    MONTHLY_BUDGET_KEY,
    REPORT_KEY,
    REPORT_SIGNATURE_KEY,
    ReportNotReadyError,
    build_report_signature,
    build_score_presentation,
    generate_financial_report,
    invalidate_review_confirmation,
    synchronize_statement_inputs,
)
from finpulse.statement_ingestion import (
    build_manual_column_mapping,
    read_statement,
)


def transaction(
    date,
    amount,
    transaction_type,
    final_category,
    *,
    predicted_category=None,
    confidence="High",
    include_in_analysis=True,
):
    return {
        "date": date,
        "transaction_id": None,
        "debit": amount if transaction_type == "debit" else 0.0,
        "credit": amount if transaction_type == "credit" else 0.0,
        "balance": None,
        "description": f"{final_category} transaction",
        "amount": float(amount),
        "transaction_type": transaction_type,
        "receiver": "Test Receiver",
        "detailed_category": final_category,
        "predicted_category": predicted_category or final_category,
        "confidence": confidence,
        "final_category": final_category,
        "include_in_analysis": include_in_analysis,
    }


def reviewed_frame():
    return pd.DataFrame(
        [
            transaction("2024-01-01", 5000, "credit", "Income"),
            transaction(
                "2024-01-05",
                1000,
                "debit",
                "Essentials",
                predicted_category="Desire",
                confidence="Low",
            ),
            transaction("2024-01-10", 500, "debit", "Desire"),
            transaction("2024-01-15", 300, "debit", "Repayment"),
            transaction("2024-01-20", 400, "debit", "Investment/Savings"),
            transaction("2024-01-31", 100, "debit", "Others"),
        ]
    )


class ReportingTests(unittest.TestCase):
    def test_review_summary_counts_excluded_rows(self):
        frame = reviewed_frame()
        frame.loc[1, "include_in_analysis"] = False
        report = generate_financial_report(frame, 5000, review_confirmed=True)
        self.assertEqual(report.review_summary["transaction_count"], 6)
        self.assertEqual(report.review_summary["included_in_analysis"], 5)
        self.assertEqual(report.review_summary["excluded_from_analysis"], 1)
        self.assertEqual(len(report.excluded_transactions), 1)

    def test_inclusion_change_invalidates_confirmation_and_report(self):
        frame = reviewed_frame()
        original_signature = build_report_signature(
            file_fingerprint="file",
            reviewed_transactions=frame,
            monthly_available_amount=5000,
            monthly_budget=None,
        )
        frame.loc[1, "include_in_analysis"] = False
        changed_signature = build_report_signature(
            file_fingerprint="file",
            reviewed_transactions=frame,
            monthly_available_amount=5000,
            monthly_budget=None,
        )
        self.assertNotEqual(original_signature, changed_signature)

        state = {
            "reviewed_transactions": reviewed_frame(),
            REPORT_KEY: object(),
            ANALYTICS_KEY: object(),
            REPORT_SIGNATURE_KEY: original_signature,
        }
        invalidate_review_confirmation(state)
        self.assertNotIn("reviewed_transactions", state)
        self.assertNotIn(REPORT_KEY, state)
        self.assertNotIn(ANALYTICS_KEY, state)

    def test_segmentation_receives_only_included_derived_features(self):
        frame = reviewed_frame()
        exceptional = transaction(
            "2024-01-25",
            20000,
            "debit",
            "Desire",
            include_in_analysis=False,
        )
        frame = pd.concat([frame, pd.DataFrame([exceptional])], ignore_index=True)
        with mock.patch(
            "finpulse.reporting.predict_persona", return_value=object()
        ) as predictor:
            generate_financial_report(frame, 5000, review_confirmed=True)
        analytics_passed_to_segmentation = predictor.call_args.args[0]
        self.assertEqual(
            analytics_passed_to_segmentation.behavioral_features[
                "avg_desire_ratio"
            ],
            0.1,
        )

    def test_analytics_uses_final_category(self):
        report = generate_financial_report(
            reviewed_frame(), 5000, review_confirmed=True
        )
        self.assertEqual(report.analytics.category_summary["Essentials"]["total"], 1000)
        self.assertEqual(report.review_summary["manually_corrected"], 1)

    def test_report_requires_confirmed_review(self):
        with self.assertRaisesRegex(ReportNotReadyError, "Confirm reviewed"):
            generate_financial_report(
                reviewed_frame(), 5000, review_confirmed=False
            )

    def test_persona_unavailable_is_a_normal_result(self):
        report = generate_financial_report(
            reviewed_frame(), 5000, review_confirmed=True
        )
        self.assertFalse(report.persona.persona_available)
        self.assertEqual(report.persona.persona_confidence, "Unavailable")
        self.assertTrue(report.persona.reasons)

    def test_upload_score_is_complete_and_confidence_is_separate(self):
        report = generate_financial_report(
            reviewed_frame(), 5000, review_confirmed=True
        )
        display = report.score_presentation
        self.assertFalse(display.is_provisional)
        self.assertEqual(display.status_label, "COMPLETE 100-POINT FRAMEWORK")
        self.assertIn("three observable", display.explanation)
        self.assertIn("Report Confidence", display.explanation)
        self.assertEqual(report.analytics.data_quality["analytical_confidence"], "Low")

    def test_score_presentation_requires_upload_component_contract(self):
        display = build_score_presentation(
            {
                "components": {
                    "spending_control": {},
                    "saving_investment": {},
                    "repayment_management": {},
                },
            }
        )
        self.assertFalse(display.is_provisional)
        self.assertEqual(display.status_label, "COMPLETE 100-POINT FRAMEWORK")
        self.assertNotIn("provisional", display.explanation.lower())

    def test_absent_monthly_budget_does_not_error(self):
        report = generate_financial_report(
            reviewed_frame(), 5000, None, review_confirmed=True
        )
        self.assertIsNone(report.analytics.budget_summary)

    def test_changing_uploaded_file_invalidates_stale_state(self):
        state = {
            FILE_FINGERPRINT_KEY: "old-file",
            MONTHLY_AMOUNT_KEY: 5000.0,
            MONTHLY_BUDGET_KEY: None,
            "categorized_transactions": reviewed_frame(),
            "reviewed_transactions": reviewed_frame(),
            REPORT_KEY: object(),
            ANALYTICS_KEY: object(),
            REPORT_SIGNATURE_KEY: "stale",
        }
        changes = synchronize_statement_inputs(
            state,
            file_fingerprint="new-file",
            monthly_available_amount=5000,
            monthly_budget=None,
        )
        self.assertEqual(changes, ("uploaded_file",))
        self.assertNotIn("reviewed_transactions", state)
        self.assertNotIn(REPORT_KEY, state)
        self.assertNotIn(ANALYTICS_KEY, state)

    def test_changing_monthly_amount_invalidates_analytics_only(self):
        reviewed = reviewed_frame()
        state = {
            FILE_FINGERPRINT_KEY: "same-file",
            MONTHLY_AMOUNT_KEY: 5000.0,
            MONTHLY_BUDGET_KEY: None,
            "categorized_transactions": reviewed.copy(),
            "reviewed_transactions": reviewed,
            REPORT_KEY: object(),
            ANALYTICS_KEY: object(),
        }
        changes = synchronize_statement_inputs(
            state,
            file_fingerprint="same-file",
            monthly_available_amount=6000,
            monthly_budget=None,
        )
        self.assertEqual(changes, ("monthly_available_amount",))
        self.assertIn("reviewed_transactions", state)
        self.assertNotIn(REPORT_KEY, state)
        self.assertNotIn(ANALYTICS_KEY, state)

    def test_changing_monthly_budget_invalidates_report_but_preserves_review(self):
        reviewed = reviewed_frame()
        state = {
            FILE_FINGERPRINT_KEY: "same-file",
            MONTHLY_AMOUNT_KEY: 5000.0,
            MONTHLY_BUDGET_KEY: 2000.0,
            "reviewed_transactions": reviewed,
            REPORT_KEY: object(),
            ANALYTICS_KEY: object(),
        }
        changes = synchronize_statement_inputs(
            state,
            file_fingerprint="same-file",
            monthly_available_amount=5000,
            monthly_budget=2500,
        )
        self.assertEqual(changes, ("monthly_budget",))
        self.assertIn("reviewed_transactions", state)
        self.assertNotIn(REPORT_KEY, state)
        self.assertNotIn(ANALYTICS_KEY, state)

    def test_credit_income_is_not_counted_as_debit_spending(self):
        report = generate_financial_report(
            reviewed_frame(), 5000, review_confirmed=True
        )
        income_row = report.debit_spending_breakdown.set_index("category").loc[
            "Income"
        ]
        self.assertEqual(income_row["total"], 0)
        self.assertEqual(
            report.analytics.statement_cash_flow_summary["observed_credits"],
            5000,
        )
        self.assertTrue(
            (report.debit_spending_breakdown["cash_flow"] == "debit").all()
        )

    def test_income_labelled_debit_remains_in_spending_reconciliation(self):
        frame = reviewed_frame()
        frame.loc[1, "final_category"] = "Income"
        report = generate_financial_report(frame, 5000, review_confirmed=True)
        self.assertEqual(
            report.debit_spending_breakdown["total"].sum(),
            report.analytics.statement_summary["total_debit_spending"],
        )
        income_row = report.debit_spending_breakdown.set_index("category").loc[
            "Income"
        ]
        self.assertEqual(income_row["total"], 1000)
        self.assertEqual(report.behavioral_ratios["Income-labelled debit"], 0.2)

    def test_six_category_breakdown_uses_final_categories(self):
        report = generate_financial_report(
            reviewed_frame(), 5000, review_confirmed=True
        )
        self.assertEqual(
            report.category_breakdown["category"].tolist(),
            [
                "Income",
                "Essentials",
                "Desire",
                "Repayment",
                "Investment/Savings",
                "Others",
            ],
        )
        essentials = report.category_breakdown.set_index("category").loc[
            "Essentials", "total"
        ]
        self.assertEqual(essentials, 1000)

    def test_report_generation_performs_no_database_access(self):
        with mock.patch.object(
            sqlite3,
            "connect",
            side_effect=AssertionError("database access is forbidden"),
        ):
            report = generate_financial_report(
                reviewed_frame(), 5000, review_confirmed=True
            )
        self.assertEqual(report.review_summary["transaction_count"], 6)

    def test_manual_mapping_resolves_ambiguous_required_field(self):
        data = (
            "Date,Value Date,Details,Withdraw,Deposit\n"
            "01/01/2024,02/01/2024,Groceries,100,\n"
        ).encode("utf-8")
        automatic = read_statement(data, filename="ambiguous.csv")
        self.assertFalse(automatic.is_usable)

        mapping = build_manual_column_mapping(
            automatic.diagnostics.source_columns,
            date="Value Date",
            description="Details",
            debit="Withdraw",
            credit="Deposit",
        )
        resolved = read_statement(
            data,
            filename="ambiguous.csv",
            mapping=mapping,
        )
        self.assertTrue(resolved.is_usable)
        self.assertEqual(
            resolved.transactions.loc[0, "date"].date().isoformat(),
            "2024-01-02",
        )


if __name__ == "__main__":
    unittest.main()
