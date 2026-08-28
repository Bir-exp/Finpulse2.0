import json
from pathlib import Path
import re
import unittest

import pandas as pd

from finpulse.categorization import ALLOWED_CATEGORIES, categorize_transactions
from finpulse.reporting import generate_financial_report
from finpulse.review import apply_category_edits, initialize_review_categories
from finpulse.statement_ingestion import read_statement


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "statements"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
EXPECTED_FIXTURES = {
    "student_1month.xlsx",
    "salaried_3months.xlsx",
    "household_3months.csv",
    "mixed_6months.xlsx",
    "alternate_bank_format.csv",
    "amount_type_format.xlsx",
}
DEBIT_CATEGORIES = (
    "Income",
    "Essentials",
    "Desire",
    "Repayment",
    "Investment/Savings",
    "Others",
)


class StatementFixtureCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.specs = {
            item["filename"]: item for item in cls.manifest["fixtures"]
        }

    def load_pipeline(self, filename):
        path = FIXTURE_DIR / filename
        ingestion = read_statement(path.read_bytes(), filename=filename)
        categorized = categorize_transactions(ingestion.transactions)
        reviewed = initialize_review_categories(categorized)
        return ingestion, categorized, reviewed

    def apply_manifest_exclusions(self, reviewed, spec):
        result = reviewed.copy()
        ids = {
            item["transaction_id"]
            for item in spec["exceptional_transactions"]
            if item.get("intended_exclusion")
        }
        if ids:
            result.loc[result["transaction_id"].isin(ids), "include_in_analysis"] = False
        return result

    def assert_report_reconciles(self, report):
        cash = report.analytics.statement_cash_flow_summary
        self.assertAlmostEqual(
            cash["net_statement_cash_flow"],
            cash["observed_credits"] - cash["observed_debits"],
            places=2,
        )
        category_total = sum(
            report.analytics.category_summary[category]["total"]
            for category in DEBIT_CATEGORIES
        )
        self.assertAlmostEqual(
            category_total,
            cash["spending_considered_for_analysis"],
            places=2,
        )
        self.assertAlmostEqual(
            report.debit_spending_breakdown["total"].sum(),
            cash["spending_considered_for_analysis"],
            places=2,
        )

    def test_manifest_declares_complete_privacy_safe_fixture_set(self):
        self.assertEqual(set(self.specs), EXPECTED_FIXTURES)
        self.assertIn("synthetic", self.manifest["privacy_notice"].casefold())
        self.assertIn("No real user", self.manifest["privacy_notice"])
        self.assertEqual(
            self.manifest["preferred_demo"]["filename"],
            "student_1month.xlsx",
        )
        for spec in self.specs.values():
            self.assertGreater(spec["monthly_available_amount"], 0)
            self.assertGreater(spec["expected_approx_transaction_count"], 0)
            self.assertIn("notes", spec)

    def test_every_fixture_ingests_maps_categorizes_and_reports(self):
        observed_categories = set()
        for filename, spec in self.specs.items():
            with self.subTest(filename=filename):
                ingestion, categorized, reviewed = self.load_pipeline(filename)
                self.assertTrue(ingestion.is_usable)
                self.assertEqual(
                    ingestion.diagnostics.original_row_count,
                    spec["expected_source_row_count"],
                )
                self.assertEqual(
                    ingestion.diagnostics.cleaned_row_count,
                    spec["expected_cleaned_row_count"],
                )
                self.assertLessEqual(
                    abs(len(categorized) - spec["expected_approx_transaction_count"]),
                    2,
                )
                self.assertEqual(
                    list(ingestion.diagnostics.date_range),
                    spec["expected_date_range"],
                )
                self.assertEqual(
                    ingestion.diagnostics.detected_column_mapping,
                    spec["expected_column_mapping"],
                )
                self.assertEqual(
                    ingestion.mapping_result.amount_layout,
                    spec["expected_amount_layout"],
                )
                self.assertEqual(
                    ingestion.diagnostics.dropped_row_reasons,
                    spec["expected_dropped_row_reasons"],
                )
                self.assertFalse(spec["manual_mapping_required"])
                self.assertTrue(
                    set(categorized["predicted_category"]).issubset(
                        ALLOWED_CATEGORIES
                    )
                )
                observed_categories.update(categorized["predicted_category"])
                self.assertTrue(
                    reviewed["final_category"].equals(
                        reviewed["predicted_category"].astype("string")
                    )
                )
                self.assertTrue(reviewed["include_in_analysis"].all())
                if spec["low_confidence_expected"]:
                    self.assertTrue((categorized["confidence"] == "Low").any())

                for expectation in spec["category_expectations"]:
                    if "transaction_id" in expectation:
                        rows = categorized.loc[
                            categorized["transaction_id"].eq(
                                expectation["transaction_id"]
                            )
                        ]
                    else:
                        rows = categorized.loc[
                            categorized["description"].eq(expectation["description"])
                        ]
                    self.assertFalse(rows.empty, expectation)
                    self.assertTrue(
                        rows["predicted_category"].eq(
                            expectation["predicted_category"]
                        ).all(),
                        expectation,
                    )
                    if "confidence" in expectation:
                        self.assertTrue(
                            rows["confidence"].eq(expectation["confidence"]).all(),
                            expectation,
                        )

                reviewed = self.apply_manifest_exclusions(reviewed, spec)
                report = generate_financial_report(
                    reviewed,
                    spec["monthly_available_amount"],
                    spec["monthly_budget"],
                    review_confirmed=True,
                )
                self.assert_report_reconciles(report)
                self.assertTrue(report.score_presentation.is_provisional)
                self.assertFalse(
                    report.analytics.score_result["components"]["stability"][
                        "available"
                    ]
                )
                self.assertFalse(report.persona.persona_available)
                if spec["monthly_budget"] is None:
                    self.assertIsNone(report.analytics.budget_summary)
                else:
                    budget = report.analytics.budget_summary
                    self.assertAlmostEqual(
                        budget["budget_utilization"],
                        budget["monthly_normalized_debit_spending"]
                        / spec["monthly_budget"],
                        places=4,
                    )
        self.assertEqual(observed_categories, set(ALLOWED_CATEGORIES))

    def test_detailed_category_coverage(self):
        observed = set()
        for filename in self.specs:
            _, categorized, _ = self.load_pipeline(filename)
            observed.update(categorized["detailed_category"])
        expected = {
            "Groceries",
            "Healthcare",
            "Utilities",
            "Rent",
            "Education/Stationery",
            "Food Delivery",
            "Food/Canteen",
            "Entertainment",
            "OTT/Subscriptions",
            "Shopping/Fashion",
            "Personal Care",
            "Transport/Fuel",
            "Credit Card Payment",
            "Loan EMI",
            "Mutual Fund/SIP",
            "Unknown Transfer",
        }
        self.assertTrue(expected.issubset(observed), expected - observed)

    def test_household_dirty_rows_and_optional_columns(self):
        ingestion, _, _ = self.load_pipeline("household_3months.csv")
        self.assertEqual(ingestion.diagnostics.dropped_row_reasons, {
            "duplicate_header_row": 1,
            "blank_row": 1,
            "missing_amount_direction": 1,
        })
        self.assertTrue(ingestion.transactions["transaction_id"].isna().all())
        self.assertTrue(ingestion.transactions["balance"].isna().all())
        self.assertEqual(
            ingestion.transactions["date"].min().date().isoformat(),
            "2026-07-01",
        )
        self.assertIn(1250.0, ingestion.transactions["amount"].tolist())

    def test_amount_type_and_alternate_alias_formats(self):
        amount_ingestion, _, _ = self.load_pipeline("amount_type_format.xlsx")
        self.assertEqual(
            amount_ingestion.mapping_result.amount_layout,
            "amount_and_transaction_type",
        )
        self.assertEqual(
            set(amount_ingestion.transactions["transaction_type"]),
            {"debit", "credit"},
        )
        alternate_ingestion, _, _ = self.load_pipeline("alternate_bank_format.csv")
        self.assertEqual(
            alternate_ingestion.diagnostics.detected_column_mapping["debit"],
            "Withdrawal Amount",
        )
        self.assertEqual(
            alternate_ingestion.diagnostics.detected_column_mapping["credit"],
            "Deposit Amount",
        )

    def test_repeated_ids_are_preserved_for_review(self):
        ingestion, _, _ = self.load_pipeline("mixed_6months.xlsx")
        repeated = ingestion.transactions.loc[
            ingestion.transactions["transaction_id"].eq("MIX-DUP-001")
        ]
        self.assertEqual(len(repeated), 2)

    def test_configured_exclusions_match_removing_exceptional_rows(self):
        for filename in ("student_1month.xlsx", "mixed_6months.xlsx"):
            with self.subTest(filename=filename):
                spec = self.specs[filename]
                _, _, reviewed = self.load_pipeline(filename)
                ids = {
                    item["transaction_id"]
                    for item in spec["exceptional_transactions"]
                }
                excluded = self.apply_manifest_exclusions(reviewed, spec)
                removed = reviewed.loc[~reviewed["transaction_id"].isin(ids)].copy()
                excluded_report = generate_financial_report(
                    excluded,
                    spec["monthly_available_amount"],
                    spec["monthly_budget"],
                    review_confirmed=True,
                )
                removed_report = generate_financial_report(
                    removed,
                    spec["monthly_available_amount"],
                    spec["monthly_budget"],
                    review_confirmed=True,
                )
                self.assertEqual(
                    excluded_report.analytics.behavioral_features,
                    removed_report.analytics.behavioral_features,
                )
                self.assertEqual(
                    excluded_report.analytics.statement_cash_flow_summary[
                        "spending_considered_for_analysis"
                    ],
                    removed_report.analytics.statement_cash_flow_summary[
                        "spending_considered_for_analysis"
                    ],
                )
                self.assertGreater(
                    excluded_report.analytics.statement_cash_flow_summary[
                        "observed_debits"
                    ],
                    removed_report.analytics.statement_cash_flow_summary[
                        "observed_debits"
                    ],
                )

    def test_adding_credit_does_not_change_behavioral_outputs(self):
        for filename, spec in self.specs.items():
            with self.subTest(filename=filename):
                _, _, reviewed = self.load_pipeline(filename)
                reviewed = self.apply_manifest_exclusions(reviewed, spec)
                baseline = generate_financial_report(
                    reviewed,
                    spec["monthly_available_amount"],
                    spec["monthly_budget"],
                    review_confirmed=True,
                )
                synthetic_credit = reviewed.iloc[[0]].copy()
                synthetic_credit.index = [max(reviewed.index) + 1000]
                synthetic_credit["transaction_id"] = "FIXTURE-CREDIT-INVARIANCE"
                synthetic_credit["debit"] = 0.0
                synthetic_credit["credit"] = 9_999_999.0
                synthetic_credit["amount"] = 9_999_999.0
                synthetic_credit["transaction_type"] = "credit"
                synthetic_credit["description"] = "IRREGULAR SYNTHETIC CREDIT"
                synthetic_credit["receiver"] = "Synthetic Credit"
                synthetic_credit["detailed_category"] = "Unknown Transfer"
                synthetic_credit["predicted_category"] = "Others"
                synthetic_credit["final_category"] = "Others"
                synthetic_credit["confidence"] = "Low"
                synthetic_credit["include_in_analysis"] = True
                extended = pd.concat([reviewed, synthetic_credit])
                changed = generate_financial_report(
                    extended,
                    spec["monthly_available_amount"],
                    spec["monthly_budget"],
                    review_confirmed=True,
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
                    baseline.persona.feature_values,
                    changed.persona.feature_values,
                )
                self.assertGreater(
                    changed.analytics.statement_cash_flow_summary["observed_credits"],
                    baseline.analytics.statement_cash_flow_summary["observed_credits"],
                )

    def test_no_obvious_real_personal_identifiers(self):
        forbidden_patterns = {
            "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
            "phone": re.compile(r"\b[6-9]\d{9}\b"),
            "public_email": re.compile(
                r"@[a-z0-9.-]*(?:gmail|yahoo|outlook|hotmail)\.[a-z]{2,}",
                flags=re.IGNORECASE,
            ),
            "account_label": re.compile(
                r"\b(?:account\s*(?:number|no)|aadhaar|passport)\b",
                flags=re.IGNORECASE,
            ),
        }
        for filename in self.specs:
            with self.subTest(filename=filename):
                ingestion, _, _ = self.load_pipeline(filename)
                text = " ".join(
                    [
                        str(value)
                        for value in ingestion.transactions["description"].tolist()
                        + ingestion.transactions["transaction_id"].tolist()
                    ]
                )
                for label, pattern in forbidden_patterns.items():
                    self.assertIsNone(pattern.search(text), label)

    def test_preferred_demo_benchmark(self):
        demo = self.manifest["preferred_demo"]
        spec = self.specs[demo["filename"]]
        _, _, reviewed = self.load_pipeline(demo["filename"])
        for correction in demo["manual_corrections"]:
            index = reviewed.index[
                reviewed["transaction_id"].eq(correction["transaction_id"])
            ][0]
            reviewed = apply_category_edits(
                reviewed, {index: correction["final_category"]}
            )
        reviewed.loc[
            reviewed["transaction_id"].isin(demo["exclusions"]),
            "include_in_analysis",
        ] = False
        report = generate_financial_report(
            reviewed,
            demo["monthly_available_amount"],
            demo["monthly_budget"],
            review_confirmed=True,
        )
        expected = demo["expected_results"]
        cash = report.analytics.statement_cash_flow_summary
        self.assertEqual(
            cash["spending_considered_for_analysis"],
            expected["spending_considered_for_analysis"],
        )
        self.assertEqual(
            {
                category: summary["total"]
                for category, summary in report.analytics.category_summary.items()
            },
            expected["category_totals"],
        )
        self.assertEqual(
            report.analytics.behavioral_features["avg_expense_ratio"],
            expected["expense_ratio"],
        )
        self.assertEqual(
            report.analytics.budget_summary["budget_utilization"],
            expected["budget_utilization"],
        )
        self.assertEqual(
            report.analytics.budget_summary["budget_utilization_percent"],
            expected["budget_utilization_percent"],
        )
        self.assert_report_reconciles(report)


if __name__ == "__main__":
    unittest.main()
