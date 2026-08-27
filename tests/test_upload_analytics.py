import sqlite3
import unittest
from unittest import mock

import pandas as pd

from finpulse.segmentation_inference import FEATURE_NAMES
from finpulse.upload_analytics import analyze_reviewed_transactions


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


def normal_month_rows():
    return [
        transaction("2024-01-01", 5000, "credit", "Income"),
        transaction("2024-01-05", 1000, "debit", "Essentials"),
        transaction("2024-01-10", 500, "debit", "Desire"),
        transaction("2024-01-15", 300, "debit", "Repayment"),
        transaction("2024-01-20", 400, "debit", "Investment/Savings"),
        transaction("2024-01-31", 100, "debit", "Others"),
    ]


def repeated_full_months(month_count=6):
    rows = []
    periods = pd.period_range("2024-01", periods=month_count, freq="M")
    for position, period in enumerate(periods):
        rows.extend([
            transaction(period.start_time, 5000 + position * 100, "credit", "Income"),
            transaction(period.start_time, 1200 + position * 50, "debit", "Essentials"),
            transaction(period.start_time + pd.Timedelta(days=10), 300 + position * 30, "debit", "Desire"),
        ])
    rows.append(transaction(periods[-1].end_time.normalize(), 100, "debit", "Others"))
    return rows


class UploadAnalyticsTests(unittest.TestCase):
    def test_large_credit_does_not_change_behavioral_outputs(self):
        debit_rows = [
            transaction("2024-01-01", 1000, "debit", "Essentials"),
            transaction("2024-01-10", 500, "debit", "Desire"),
            transaction("2024-01-20", 300, "debit", "Repayment"),
            transaction("2024-01-31", 400, "debit", "Investment/Savings"),
        ]
        with_credit = debit_rows + [
            transaction("2024-01-15", 1_000_000, "credit", "Income")
        ]
        baseline = analyze_reviewed_transactions(pd.DataFrame(debit_rows), 5000)
        credited = analyze_reviewed_transactions(pd.DataFrame(with_credit), 5000)

        ratio_features = (
            "avg_essential_ratio",
            "avg_desire_ratio",
            "avg_repayment_ratio",
            "avg_investment_ratio",
            "avg_other_ratio",
            "avg_expense_ratio",
            "avg_surplus_ratio",
        )
        for feature in ratio_features:
            self.assertEqual(
                baseline.behavioral_features[feature],
                credited.behavioral_features[feature],
            )
        self.assertEqual(baseline.score_result, credited.score_result)
        self.assertEqual(baseline.signals, credited.signals)
        self.assertEqual(baseline.recommendations, credited.recommendations)
        self.assertEqual(
            {name: baseline.behavioral_features[name] for name in FEATURE_NAMES},
            {name: credited.behavioral_features[name] for name in FEATURE_NAMES},
        )

    def test_credits_remain_visible_in_statement_cash_flow(self):
        rows = [
            transaction("2024-01-01", 5000, "credit", "Income"),
            transaction("2024-01-31", 1200, "debit", "Essentials"),
        ]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        cash_flow = result.statement_cash_flow_summary
        self.assertEqual(cash_flow["observed_credits"], 5000)
        self.assertEqual(cash_flow["observed_debits"], 1200)
        self.assertEqual(cash_flow["net_statement_cash_flow"], 3800)
        self.assertEqual(cash_flow["spending_considered_for_analysis"], 1200)

    def test_excluded_credit_remains_visible_only_in_raw_cash_flow(self):
        rows = [
            transaction(
                "2024-01-01",
                20000,
                "credit",
                "Income",
                include_in_analysis=False,
            ),
            transaction("2024-01-31", 1000, "debit", "Essentials"),
        ]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        self.assertEqual(result.statement_cash_flow_summary["observed_credits"], 20000)
        self.assertEqual(result.observed_income_summary["total_credit_inflow"], 0)
        self.assertIsNone(result.behavioral_features["income_cv"])

    def test_upload_stability_is_unavailable_even_with_long_credit_history(self):
        rows = repeated_full_months(6)
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        stability = result.score_result["components"]["stability"]
        self.assertFalse(stability["available"])
        self.assertIsNone(stability["score"])
        self.assertIsNone(result.behavioral_features["income_cv"])
        self.assertTrue(result.score_result["is_provisional"])
        self.assertIn("bank credits", stability["unavailable_reason"])

    def test_excluded_debit_does_not_count_toward_spending(self):
        rows = [
            transaction("2024-01-01", 1000, "debit", "Essentials"),
            transaction(
                "2024-01-31",
                20000,
                "debit",
                "Others",
                include_in_analysis=False,
            ),
        ]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        self.assertEqual(result.statement_summary["total_debit_spending"], 1000)
        self.assertEqual(result.statement_summary["original_reviewed_transaction_count"], 2)
        self.assertEqual(result.statement_summary["included_transaction_count"], 1)
        self.assertEqual(result.statement_summary["excluded_transaction_count"], 1)
        self.assertEqual(
            result.statement_cash_flow_summary["spending_considered_for_analysis"],
            1000,
        )
        self.assertEqual(result.statement_cash_flow_summary["observed_debits"], 21000)

    def test_excluded_desire_does_not_affect_ratio(self):
        rows = [
            transaction("2024-01-01", 500, "debit", "Desire"),
            transaction(
                "2024-01-31",
                20000,
                "debit",
                "Desire",
                include_in_analysis=False,
            ),
        ]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        self.assertEqual(result.behavioral_features["avg_desire_ratio"], 0.1)

    def test_excluded_repayment_does_not_affect_ratio(self):
        rows = [
            transaction("2024-01-01", 250, "debit", "Repayment"),
            transaction(
                "2024-01-31",
                20000,
                "debit",
                "Repayment",
                include_in_analysis=False,
            ),
        ]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        self.assertEqual(result.behavioral_features["avg_repayment_ratio"], 0.05)

    def test_excluded_investment_does_not_affect_ratio(self):
        rows = [
            transaction("2024-01-01", 500, "debit", "Investment/Savings"),
            transaction(
                "2024-01-31",
                20000,
                "debit",
                "Investment/Savings",
                include_in_analysis=False,
            ),
        ]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        self.assertEqual(result.behavioral_features["avg_investment_ratio"], 0.1)

    def test_excluded_credit_does_not_affect_income_or_stability(self):
        rows = [
            transaction("2024-01-01", 5000, "credit", "Income"),
            transaction(
                "2024-01-15",
                20000,
                "credit",
                "Income",
                include_in_analysis=False,
            ),
            transaction("2024-01-31", 100, "debit", "Essentials"),
            transaction("2024-02-01", 5000, "credit", "Income"),
            transaction("2024-02-29", 100, "debit", "Essentials"),
        ]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        self.assertEqual(result.observed_income_summary["observed_income_credits"], 10000)
        self.assertEqual(result.observed_income_summary["total_credit_inflow"], 10000)
        self.assertIsNone(result.behavioral_features["income_cv"])
        self.assertFalse(result.data_quality["stability_features_available"])
        self.assertEqual(result.statement_cash_flow_summary["observed_credits"], 30000)

    def test_explicitly_included_rows_match_legacy_behavior(self):
        included = pd.DataFrame(normal_month_rows())
        legacy = included.drop(columns="include_in_analysis")
        included_result = analyze_reviewed_transactions(included, 5000)
        legacy_result = analyze_reviewed_transactions(legacy, 5000)
        self.assertEqual(included_result.category_summary, legacy_result.category_summary)
        self.assertEqual(
            included_result.behavioral_features,
            legacy_result.behavioral_features,
        )

    def test_confidence_uses_included_transaction_count(self):
        rows = []
        for position in range(20):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(
                days=round(position * 30 / 19)
            )
            rows.append(transaction(date, 10, "debit", "Essentials"))
            rows.append(
                transaction(
                    date,
                    10,
                    "debit",
                    "Others",
                    include_in_analysis=False,
                )
            )
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        self.assertEqual(result.data_quality["original_reviewed_transaction_count"], 40)
        self.assertEqual(result.data_quality["included_transaction_count"], 20)
        self.assertEqual(result.data_quality["analytical_confidence"], "Low")
        self.assertFalse(result.data_quality["minimum_transaction_guidance_met"])

    def test_excluding_every_transaction_fails_clearly(self):
        rows = [
            transaction(
                "2024-01-01",
                100,
                "debit",
                "Essentials",
                include_in_analysis=False,
            )
        ]
        with self.assertRaisesRegex(ValueError, "No transactions are included"):
            analyze_reviewed_transactions(pd.DataFrame(rows), 5000)

    def test_one_month_statement(self):
        result = analyze_reviewed_transactions(
            pd.DataFrame(normal_month_rows()),
            monthly_available_amount=5000,
        )
        self.assertEqual(result.statement_summary["coverage_days"], 27)
        self.assertEqual(result.statement_summary["normalization_months"], 1.0)
        self.assertEqual(result.behavioral_features["avg_essential_ratio"], 0.2)

    def test_multi_month_statement_normalizes_monthly(self):
        rows = normal_month_rows()
        second = []
        for row in normal_month_rows():
            copied = dict(row)
            copied["date"] = pd.Timestamp(row["date"]) + pd.DateOffset(months=1)
            second.append(copied)
        frame = pd.DataFrame(rows + second)

        result = analyze_reviewed_transactions(frame, 5000)

        self.assertEqual(result.statement_summary["normalization_months"], 1.871)
        self.assertEqual(
            result.category_summary["Essentials"]["monthly_normalized"],
            1068.97,
        )
        self.assertEqual(result.behavioral_features["avg_essential_ratio"], 0.2138)

    def test_short_statement_has_low_analytical_confidence(self):
        rows = [
            transaction("2024-01-01", 100, "debit", "Essentials"),
            transaction("2024-01-10", 1000, "credit", "Income"),
        ]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        self.assertEqual(result.data_quality["analytical_confidence"], "Low")
        self.assertFalse(result.data_quality["minimum_coverage_guidance_met"])

    def test_fewer_than_30_transactions_lowers_confidence(self):
        result = analyze_reviewed_transactions(
            pd.DataFrame(normal_month_rows()), 5000
        )
        self.assertEqual(result.data_quality["analytical_confidence"], "Low")
        self.assertFalse(result.data_quality["minimum_transaction_guidance_met"])

    def test_final_category_correction_affects_analytics(self):
        row = transaction(
            "2024-01-01",
            1000,
            "debit",
            "Essentials",
            predicted_category="Desire",
        )
        result = analyze_reviewed_transactions(pd.DataFrame([row]), 5000)
        self.assertEqual(result.category_summary["Essentials"]["total"], 1000)
        self.assertEqual(result.category_summary["Desire"]["total"], 0)

    def test_predicted_category_does_not_affect_analytics(self):
        first = transaction("2024-01-01", 500, "debit", "Others", predicted_category="Desire")
        second = dict(first, predicted_category="Essentials")
        result_a = analyze_reviewed_transactions(pd.DataFrame([first]), 5000)
        result_b = analyze_reviewed_transactions(pd.DataFrame([second]), 5000)
        self.assertEqual(result_a.category_summary, result_b.category_summary)
        self.assertEqual(result_a.behavioral_features, result_b.behavioral_features)

    def test_monthly_available_amount_is_ratio_denominator(self):
        rows = [transaction("2024-01-01", 1000, "debit", "Essentials")]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 4000)
        self.assertEqual(result.behavioral_features["avg_essential_ratio"], 0.25)

    def test_observed_credits_do_not_replace_reference_amount(self):
        rows = [
            transaction("2024-01-01", 100000, "credit", "Income"),
            transaction("2024-01-31", 1000, "debit", "Essentials"),
        ]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        self.assertEqual(result.behavioral_features["avg_essential_ratio"], 0.2)
        self.assertFalse(
            result.observed_income_summary["reference_amount_was_replaced_by_credits"]
        )

    def test_optional_budget_none(self):
        result = analyze_reviewed_transactions(
            pd.DataFrame(normal_month_rows()), 5000, None
        )
        self.assertIsNone(result.budget_summary)

    def test_budget_utilization(self):
        rows = [transaction("2024-01-01", 1000, "debit", "Essentials")]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000, 2000)
        self.assertEqual(result.budget_summary["budget_utilization"], 0.5)
        self.assertEqual(result.budget_summary["budget_variance"], 1000)

    def test_income_credits_do_not_count_as_debit_spending(self):
        rows = [
            transaction("2024-01-01", 5000, "credit", "Income"),
            transaction("2024-01-31", 1000, "debit", "Essentials"),
        ]
        result = analyze_reviewed_transactions(pd.DataFrame(rows), 5000)
        self.assertEqual(result.statement_summary["total_debit_spending"], 1000)
        self.assertEqual(result.category_summary["Income"]["total"], 0)
        self.assertEqual(result.statement_cash_flow_summary["observed_credits"], 5000)

    def test_category_aggregates(self):
        result = analyze_reviewed_transactions(
            pd.DataFrame(normal_month_rows()), 5000
        )
        expected = {
            "Essentials": 1000,
            "Desire": 500,
            "Repayment": 300,
            "Investment/Savings": 400,
        }
        for category, amount in expected.items():
            self.assertEqual(result.category_summary[category]["total"], amount)

    def test_others_is_retained(self):
        result = analyze_reviewed_transactions(
            pd.DataFrame(normal_month_rows()), 5000
        )
        self.assertEqual(result.category_summary["Others"]["total"], 100)

    def test_optional_balance_and_transaction_id_are_not_required(self):
        frame = pd.DataFrame(normal_month_rows()).drop(
            columns=["balance", "transaction_id"]
        )
        result = analyze_reviewed_transactions(frame, 5000)
        self.assertEqual(result.statement_summary["transaction_count"], 5)
        self.assertEqual(
            result.statement_summary["original_reviewed_transaction_count"],
            len(frame),
        )

    def test_score_is_within_valid_range(self):
        result = analyze_reviewed_transactions(
            pd.DataFrame(normal_month_rows()), 5000
        )
        self.assertGreaterEqual(result.score_result["finpulse_score"], 0)
        self.assertLessEqual(result.score_result["finpulse_score"], 100)

    def test_score_contains_explainable_components(self):
        result = analyze_reviewed_transactions(
            pd.DataFrame(normal_month_rows()), 5000
        )
        self.assertEqual(
            set(result.score_result["components"]),
            {"spending_control", "savings", "debt_management", "stability"},
        )
        self.assertFalse(result.score_result["components"]["stability"]["available"])
        self.assertTrue(result.score_result["renormalized_for_unavailable_components"])

    def test_short_history_suppresses_trend_and_stability_claims(self):
        result = analyze_reviewed_transactions(
            pd.DataFrame(normal_month_rows()), 5000
        )
        self.assertFalse(result.data_quality["trend_features_available"])
        self.assertFalse(result.data_quality["stability_features_available"])
        self.assertIsNone(result.behavioral_features["desire_ratio_change_3m"])
        self.assertIsNone(result.behavioral_features["income_cv"])
        self.assertFalse(any("rising" in signal.lower() for signal in result.signals))
        self.assertFalse(any("recently" in rec["recommendation"].lower() for rec in result.recommendations))

    def test_six_complete_months_enable_debit_trends_but_not_stability(self):
        frame = pd.DataFrame(repeated_full_months(6))
        result = analyze_reviewed_transactions(frame, 5000)
        self.assertFalse(result.data_quality["stability_features_available"])
        self.assertTrue(result.data_quality["trend_features_available"])
        self.assertIsNone(result.behavioral_features["income_cv"])
        self.assertIsNone(result.behavioral_features["income_change_3m"])
        self.assertIsNotNone(result.behavioral_features["desire_ratio_change_3m"])
        self.assertFalse(result.score_result["components"]["stability"]["available"])
        self.assertTrue(result.score_result["is_provisional"])

    def test_empty_input_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "at least one transaction"):
            analyze_reviewed_transactions(pd.DataFrame(), 5000)

    def test_invalid_input_fails_clearly(self):
        frame = pd.DataFrame([{"date": "2024-01-01", "amount": 100}])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            analyze_reviewed_transactions(frame, 5000)

    def test_no_database_writes_or_connections(self):
        frame = pd.DataFrame(normal_month_rows())
        with mock.patch.object(
            sqlite3,
            "connect",
            side_effect=AssertionError("database access is forbidden"),
        ):
            result = analyze_reviewed_transactions(frame, 5000)
        self.assertEqual(result.statement_summary["transaction_count"], 5)
        self.assertEqual(
            result.statement_summary["original_reviewed_transaction_count"],
            len(frame),
        )


if __name__ == "__main__":
    unittest.main()
