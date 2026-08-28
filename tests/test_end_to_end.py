from io import BytesIO
from pathlib import Path
import sqlite3
import unittest
from unittest import mock

import pandas as pd

from finpulse.categorization import categorize_transactions
from finpulse.reporting import generate_financial_report
from finpulse.review import apply_category_edits, initialize_review_categories
from finpulse.statement_ingestion import (
    StatementReadError,
    build_manual_column_mapping,
    read_statement,
)


FIXTURE = Path(__file__).parent / "fixtures" / "realistic_statement.csv"
DEBIT_CATEGORIES = (
    "Income",
    "Essentials",
    "Desire",
    "Repayment",
    "Investment/Savings",
    "Others",
)


def reviewed_fixture():
    ingestion = read_statement(FIXTURE.read_bytes(), filename=FIXTURE.name)
    categorized = categorize_transactions(ingestion.transactions)
    return ingestion, initialize_review_categories(categorized)


def transaction(
    date,
    amount,
    transaction_type,
    category,
    *,
    include=True,
    transaction_id=None,
):
    return {
        "date": date,
        "transaction_id": transaction_id,
        "debit": amount if transaction_type == "debit" else 0.0,
        "credit": amount if transaction_type == "credit" else 0.0,
        "balance": None,
        "description": f"{category} transaction",
        "amount": float(amount),
        "transaction_type": transaction_type,
        "receiver": "Sanitized Receiver",
        "detailed_category": category,
        "predicted_category": category,
        "confidence": "High",
        "final_category": category,
        "include_in_analysis": include,
    }


class EndToEndUploadTests(unittest.TestCase):
    def assert_reconciled(self, report):
        included_debit_total = report.analytics.statement_summary[
            "total_debit_spending"
        ]
        category_total = sum(
            report.analytics.category_summary[category]["total"]
            for category in DEBIT_CATEGORIES
        )
        self.assertAlmostEqual(category_total, included_debit_total, places=2)
        self.assertAlmostEqual(
            report.debit_spending_breakdown["total"].sum(),
            included_debit_total,
            places=2,
        )
        ratio_total = sum(
            report.analytics.behavioral_features[key]
            for key in (
                "avg_income_category_debit_ratio",
                "avg_essential_ratio",
                "avg_desire_ratio",
                "avg_repayment_ratio",
                "avg_investment_ratio",
                "avg_other_ratio",
            )
        )
        self.assertAlmostEqual(
            ratio_total,
            report.analytics.behavioral_features["avg_expense_ratio"],
            places=4,
        )

    def test_exact_real_format_runs_complete_pipeline_and_audit(self):
        ingestion, reviewed = reviewed_fixture()
        self.assertTrue(ingestion.is_usable)
        self.assertEqual(
            ingestion.diagnostics.detected_column_mapping,
            {
                "date": "Date",
                "transaction_id": "transactionId",
                "debit": "withdraw",
                "credit": "deposits",
                "balance": "balance",
                "description": "remarks",
            },
        )
        self.assertEqual(reviewed["confidence"].value_counts().to_dict(), {
            "High": 21,
            "Low": 5,
            "Medium": 4,
        })
        self.assertEqual(reviewed["predicted_category"].value_counts().to_dict(), {
            "Essentials": 12,
            "Desire": 7,
            "Others": 5,
            "Income": 2,
            "Investment/Savings": 2,
            "Repayment": 2,
        })
        receiver = reviewed.set_index("description").loc[
            "UPI/ROHAN@YBL/445566", "receiver"
        ]
        self.assertEqual(receiver, "Rohan")

        with mock.patch.object(
            sqlite3, "connect", side_effect=AssertionError("SQLite forbidden")
        ), mock.patch.object(
            pd.DataFrame,
            "to_sql",
            side_effect=AssertionError("statement persistence forbidden"),
        ):
            report = generate_financial_report(
                reviewed, 50_000, 40_000, review_confirmed=True
            )

        cash = report.analytics.statement_cash_flow_summary
        self.assertEqual(cash["observed_credits"], 50_500)
        self.assertEqual(cash["observed_debits"], 36_998)
        self.assertEqual(cash["net_statement_cash_flow"], 13_502)
        self.assertEqual(cash["spending_considered_for_analysis"], 36_998)
        self.assertEqual(
            cash["net_statement_cash_flow"],
            cash["observed_credits"] - cash["observed_debits"],
        )
        self.assertEqual(
            report.analytics.behavioral_features["monthly_reference_amount"],
            50_000,
        )
        self.assertFalse(report.score_presentation.is_provisional)
        self.assertNotIn("stability", report.analytics.score_result["components"])
        self.assertFalse(report.persona.persona_available)
        self.assertIsNotNone(report.analytics.budget_summary)
        self.assert_reconciled(report)

    def test_local_known_ambiguous_and_unknown_merchants_are_conservative(self):
        _, reviewed = reviewed_fixture()
        by_description = reviewed.set_index("description")
        expected = {
            "UPI/DMART/123456789": ("Essentials", "High"),
            "UPI/SWIGGY/987654321": ("Desire", "High"),
            "UPI/LOCAL KIRANA STORE/111111": ("Essentials", "High"),
            "RAMESH CANTEEN": ("Essentials", "Medium"),
            "UPI/STAR STATIONERY/222222": ("Essentials", "High"),
            "UPI/RAVI BARBER/333333": ("Desire", "Medium"),
            "POS/STYLE SALON": ("Desire", "Medium"),
            "APOLLO PHARMACY": ("Essentials", "High"),
            "CITY MEDICAL STORE": ("Essentials", "High"),
            "INDIAN OIL PETROL PUMP": ("Essentials", "High"),
            "NETFLIX": ("Desire", "High"),
            "SPOTIFY": ("Desire", "High"),
            "ZOMATO": ("Desire", "High"),
            "BLINKIT": ("Essentials", "High"),
            "ZEPTO": ("Essentials", "High"),
            "BIGBASKET": ("Essentials", "High"),
            "MYNTRA": ("Desire", "High"),
            "ZERODHA SIP": ("Investment/Savings", "High"),
            "GROWW INVEST TECH": ("Investment/Savings", "High"),
            "AMAZON PAY": ("Others", "Low"),
            "UPI/ROHAN@YBL/445566": ("Others", "Low"),
            "UPI/UNKNOWN CORNER SHOP/998877": ("Others", "Low"),
        }
        for description, (category, confidence) in expected.items():
            with self.subTest(description=description):
                row = by_description.loc[description]
                self.assertEqual(row["predicted_category"], category)
                self.assertEqual(row["confidence"], confidence)

    def test_manual_correction_and_multiple_exclusions_drive_report(self):
        _, reviewed = reviewed_fixture()
        amazon_index = reviewed.index[reviewed["description"] == "AMAZON PAY"][0]
        original_prediction = reviewed.loc[amazon_index, "predicted_category"]
        corrected = apply_category_edits(reviewed, {amazon_index: "Essentials"})
        self.assertEqual(corrected.loc[amazon_index, "predicted_category"], original_prediction)
        self.assertEqual(corrected.loc[amazon_index, "final_category"], "Essentials")
        excluded_indexes = corrected.index[
            corrected["description"].isin(
                ["MOBILE TRANSFER TO ANITA", "UPI/TEA STALL"]
            )
        ]
        corrected.loc[excluded_indexes, "include_in_analysis"] = False

        report = generate_financial_report(
            corrected, 50_000, review_confirmed=True
        )
        cash = report.analytics.statement_cash_flow_summary
        self.assertEqual(cash["observed_debits"], 36_998)
        self.assertEqual(cash["spending_considered_for_analysis"], 35_923)
        self.assertEqual(report.review_summary["manually_corrected"], 1)
        self.assertEqual(report.review_summary["excluded_from_analysis"], 2)
        self.assertEqual(
            report.analytics.category_summary["Essentials"]["total"], 12_930
        )
        self.assert_reconciled(report)

    def test_large_temporary_cash_flow_is_informational_when_debit_excluded(self):
        _, reviewed = reviewed_fixture()
        baseline = generate_financial_report(
            reviewed, 50_000, review_confirmed=True
        )
        exceptional = pd.DataFrame([
            transaction("2026-04-15", 10**9, "credit", "Others", include=False),
            transaction("2026-04-16", 10**9, "debit", "Others", include=False),
        ])
        extended = pd.concat([reviewed, exceptional], ignore_index=True)
        changed = generate_financial_report(
            extended, 50_000, review_confirmed=True
        )
        self.assertEqual(
            baseline.analytics.behavioral_features,
            changed.analytics.behavioral_features,
        )
        self.assertEqual(
            baseline.analytics.score_result["finpulse_score"],
            changed.analytics.score_result["finpulse_score"],
        )
        self.assertEqual(
            baseline.analytics.statement_cash_flow_summary[
                "spending_considered_for_analysis"
            ],
            changed.analytics.statement_cash_flow_summary[
                "spending_considered_for_analysis"
            ],
        )
        self.assertEqual(
            changed.analytics.statement_cash_flow_summary["observed_credits"],
            1_000_050_500,
        )
        self.assertEqual(
            changed.analytics.statement_cash_flow_summary["observed_debits"],
            1_000_036_998,
        )
        self.assert_reconciled(changed)

    def test_dirty_xlsx_mixed_dates_optional_columns_and_repeated_ids(self):
        frame = pd.DataFrame({
            "Date": [
                "01/05/2026",
                "2026-05-02",
                "3 May 2026",
                "Date",
                "",
                "04-May-2026",
                "05/05/2026",
            ],
            "transactionId": ["DUP", "DUP", "T3", "transactionId", "", "T4", "T5"],
            "withdraw": ["₹1,00,000.50", "₹1,00,000.50", "", "withdraw", "", "0", ""],
            "deposits": ["", "", "1,000", "deposits", "", "", ""],
            "balance": ["9,00,000", "8,00,000", "8,01,000", "balance", "", "8,01,000", "8,01,000"],
            "remarks": ["DMART", "DMART", "IRREGULAR CREDIT", "remarks", "", "ZERO", "BLANK AMOUNT"],
        })
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False)
        result = read_statement(buffer.getvalue(), filename="dirty.xlsx")
        self.assertEqual(result.diagnostics.cleaned_row_count, 3)
        self.assertEqual(result.transactions["transaction_id"].tolist(), ["DUP", "DUP", "T3"])
        self.assertEqual(result.transactions["amount"].tolist(), [100000.5, 100000.5, 1000.0])
        self.assertEqual(result.diagnostics.dropped_row_reasons, {
            "duplicate_header_row": 1,
            "blank_row": 1,
            "missing_amount_direction": 2,
        })
        self.assertEqual(
            result.transactions["date"].dt.date.astype(str).tolist(),
            ["2026-05-01", "2026-05-02", "2026-05-03"],
        )
        dirty_report = generate_financial_report(
            initialize_review_categories(categorize_transactions(result.transactions)),
            500_000,
            review_confirmed=True,
        )
        self.assert_reconciled(dirty_report)

        optional = pd.DataFrame({
            "Date": ["2026-05-01"],
            "withdraw": [50],
            "deposits": [""],
            "remarks": ["KIRANA"],
        })
        optional_result = read_statement(
            optional.to_csv(index=False).encode(), filename="optional.csv"
        )
        self.assertTrue(pd.isna(optional_result.transactions.loc[0, "transaction_id"]))
        self.assertTrue(pd.isna(optional_result.transactions.loc[0, "balance"]))
        optional_report = generate_financial_report(
            initialize_review_categories(
                categorize_transactions(optional_result.transactions)
            ),
            1_000,
            review_confirmed=True,
        )
        self.assert_reconciled(optional_report)

    def test_manual_mapping_fallback_and_invalid_uploads(self):
        data = (
            "Date,Value Date,Details,Money Out,Money In\n"
            "01/06/2026,02/06/2026,LOCAL GROCERY,500,\n"
        ).encode()
        automatic = read_statement(data, filename="ambiguous.csv")
        self.assertFalse(automatic.is_usable)
        mapping = build_manual_column_mapping(
            automatic.diagnostics.source_columns,
            date="Value Date",
            description="Details",
            debit="Money Out",
            credit="Money In",
        )
        resolved = read_statement(data, filename="ambiguous.csv", mapping=mapping)
        self.assertTrue(resolved.is_usable)
        self.assertEqual(resolved.transactions.loc[0, "amount"], 500)
        resolved_report = generate_financial_report(
            initialize_review_categories(
                categorize_transactions(resolved.transactions)
            ),
            5_000,
            review_confirmed=True,
        )
        self.assert_reconciled(resolved_report)
        with self.assertRaisesRegex(StatementReadError, "Unsupported statement format"):
            read_statement(b"not a pdf", filename="statement.pdf")
        with self.assertRaises(StatementReadError):
            read_statement(b"not an xlsx", filename="statement.xlsx")

    def test_student_multi_month_budget_and_confidence_cases(self):
        student = pd.DataFrame([
            transaction("2026-01-01", 5000, "credit", "Income"),
            transaction("2026-01-01", 500, "debit", "Essentials"),
            transaction("2026-01-31", 250, "debit", "Desire"),
            transaction("2026-02-01", 700, "debit", "Essentials"),
            transaction("2026-02-28", 300, "debit", "Desire"),
        ])
        report = generate_financial_report(
            student, 5_000, 2_000, review_confirmed=True
        )
        self.assertEqual(report.analytics.behavioral_features["monthly_reference_amount"], 5000)
        self.assertEqual(report.analytics.statement_summary["normalization_months"], 2)
        self.assertEqual(report.analytics.statement_summary["monthly_normalized_debit_spending"], 875)
        self.assertEqual(report.analytics.budget_summary["budget_utilization"], 0.4375)
        self.assertEqual(report.analytics.data_quality["analytical_confidence"], "Low")
        self.assertIn("Included-debit coverage is shorter than 30 days.", generate_financial_report(
            pd.DataFrame([
                transaction("2026-01-01", 50, "debit", "Essentials"),
                transaction("2026-01-10", 50, "debit", "Desire"),
            ]),
            1000,
            review_confirmed=True,
        ).analytics.data_quality["warnings"])
        self.assert_reconciled(report)

    def test_all_excluded_and_no_included_debits_fail_clearly(self):
        all_excluded = pd.DataFrame([
            transaction("2026-01-01", 100, "debit", "Essentials", include=False)
        ])
        with self.assertRaisesRegex(ValueError, "No transactions are included"):
            generate_financial_report(all_excluded, 1000, review_confirmed=True)
        credits_only = pd.DataFrame([
            transaction("2026-01-01", 1000, "credit", "Income")
        ])
        with self.assertRaisesRegex(ValueError, "No included debit transactions"):
            generate_financial_report(credits_only, 1000, review_confirmed=True)

    def test_income_labelled_debit_reconciles_and_upload_copy_uses_capacity_term(self):
        frame = pd.DataFrame([
            transaction("2026-01-01", 950, "debit", "Income"),
            transaction("2026-01-31", 100, "debit", "Desire"),
        ])
        report = generate_financial_report(frame, 1000, review_confirmed=True)
        self.assert_reconciled(report)
        self.assertEqual(
            report.analytics.category_summary["Income"]["total"], 950
        )
        all_copy = " ".join(report.analytics.signals) + " " + " ".join(
            item["recommendation"] for item in report.analytics.recommendations
        )
        self.assertIn("Monthly Available Amount", all_copy)
        self.assertNotIn("your income", all_copy.casefold())
        self.assertNotIn("relative to income", all_copy.casefold())
        self.assertTrue(any(
            "Income-labelled debit spending" in warning
            for warning in report.analytics.data_quality["warnings"]
        ))


if __name__ == "__main__":
    unittest.main()
