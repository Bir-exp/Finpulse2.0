import unittest

import pandas as pd

from finpulse.presentation import (
    MONTHLY_CATEGORY_COLUMNS,
    OTHER_INCLUDED_DEBITS,
    build_monthly_spending_view,
    build_overview_metrics,
    format_inr,
    format_month_label,
    format_percentage,
    largest_spending_category,
    monthly_spending_observations,
    monthly_spending_snapshot,
    spending_metric_label,
    spending_period_context,
)
from finpulse.upload_analytics import analyze_reviewed_transactions


class PresentationHelperTests(unittest.TestCase):
    def test_indian_currency_formatting(self):
        expected = {
            5_000: "₹5,000",
            36_998: "₹36,998",
            125_000: "₹1,25,000",
            1_250_000: "₹12,50,000",
        }
        for value, formatted in expected.items():
            with self.subTest(value=value):
                self.assertEqual(format_inr(value), formatted)
                self.assertNotIn("..", formatted)

    def test_currency_handles_decimals_negative_and_missing_values(self):
        self.assertEqual(format_inr(7_273.50), "₹7,273.5")
        self.assertEqual(format_inr(-1_250), "-₹1,250")
        self.assertEqual(format_inr(None), "Unavailable")

    def test_percentage_uses_sensible_precision(self):
        self.assertEqual(format_percentage(0.842), "84.2%")
        self.assertEqual(format_percentage(161.62, already_percent=True), "161.6%")

    def test_multi_month_overview_uses_existing_normalized_value(self):
        statement = {
            "monthly_available_amount": 50_000,
            "total_debit_spending": 90_000,
            "monthly_normalized_debit_spending": 30_000,
            "remaining_amount_estimate": 20_000,
            "transaction_month_count": 3,
            "start_date": "2026-08-01",
            "end_date": "2026-10-31",
        }
        overview = build_overview_metrics(
            statement,
            {"total_score": 72, "score_band": "Stable"},
        )
        self.assertEqual(overview["monthly_spending"], 30_000)
        self.assertNotEqual(overview["monthly_spending"], 90_000)
        self.assertEqual(overview["spending_label"], "Average Monthly Spending")
        self.assertIn("3 months", overview["period_context"])
        self.assertEqual(overview["score"], 72)
        self.assertNotIn("provisional", overview)

    def test_one_month_label_and_context_are_clear(self):
        statement = {
            "transaction_month_count": 1,
            "start_date": "2026-08-01",
            "end_date": "2026-08-31",
        }
        self.assertEqual(spending_metric_label(statement), "Monthly Spending")
        self.assertIn("Aug 2026", spending_period_context(statement))

    def test_largest_category_summary_uses_monthly_values(self):
        category_summary = {
            "Essentials": {"monthly_normalized": 2_250},
            "Desire": {"monthly_normalized": 1_250},
            "Repayment": {"monthly_normalized": 500},
            "Investment/Savings": {"monthly_normalized": 750},
            "Others": {"monthly_normalized": 250},
        }
        self.assertEqual(
            largest_spending_category(category_summary),
            ("Essentials", 2_250.0),
        )

    def test_twelve_month_view_is_complete_and_chronological(self):
        rows = []
        for month in pd.period_range("2025-07", periods=12, freq="M"):
            rows.append(
                {
                    "month": str(month),
                    "coverage_days": month.days_in_month,
                    "days_in_month": month.days_in_month,
                    "is_complete_month": True,
                    "debit_transaction_count": 1,
                    "income_category_debits": 0,
                    "essentials": 1_000,
                    "desire": 500,
                    "repayment": 250,
                    "investment_savings": 100,
                    "others": 50,
                    "total_debit_spending": 1_900,
                }
            )
        view = build_monthly_spending_view(pd.DataFrame(rows), 2_000)
        self.assertEqual(len(view), 12)
        self.assertEqual(view.iloc[0]["month_key"], "2025-07")
        self.assertEqual(view.iloc[-1]["month_key"], "2026-06")
        self.assertEqual(
            view["month_period"].tolist(),
            sorted(view["month_period"].tolist()),
        )
        self.assertEqual(format_month_label("2026-06"), "June 2026")

    def test_monthly_view_uses_only_reviewed_included_debits(self):
        transactions = pd.DataFrame(
            [
                {
                    "date": "2026-01-15",
                    "amount": 100,
                    "transaction_type": "debit",
                    "predicted_category": "Essentials",
                    "final_category": "Essentials",
                    "include_in_analysis": True,
                },
                {
                    "date": "2026-01-20",
                    "amount": 200,
                    "transaction_type": "debit",
                    "predicted_category": "Desire",
                    "final_category": "Desire",
                    "include_in_analysis": False,
                },
                {
                    "date": "2026-01-25",
                    "amount": 10_000,
                    "transaction_type": "credit",
                    "predicted_category": "Income",
                    "final_category": "Income",
                    "include_in_analysis": True,
                },
                {
                    "date": "2026-02-02",
                    "amount": 300,
                    "transaction_type": "debit",
                    "predicted_category": "Essentials",
                    "final_category": "Desire",
                    "include_in_analysis": True,
                },
                {
                    "date": "2026-02-10",
                    "amount": 50,
                    "transaction_type": "debit",
                    "predicted_category": "Others",
                    "final_category": "Others",
                    "include_in_analysis": True,
                },
            ]
        )
        analytics = analyze_reviewed_transactions(transactions, 1_000, 500)
        view = build_monthly_spending_view(analytics.monthly_summary, 500)

        january = monthly_spending_snapshot(view, "2026-01")
        february = monthly_spending_snapshot(view, "February 2026 (partial)")
        self.assertEqual(january["Monthly Spending"], 100)
        self.assertEqual(january["Essentials"], 100)
        self.assertEqual(january["Desire"], 0)
        self.assertEqual(january["Budget Used"], 0.2)
        self.assertTrue(january["Partial Month"])
        self.assertEqual(january["Coverage Days"], 17)

        self.assertEqual(february["Monthly Spending"], 350)
        self.assertEqual(february["Essentials"], 0)
        self.assertEqual(february["Desire"], 300)
        self.assertEqual(february["Budget Used"], 0.7)
        for _, month in view.iterrows():
            category_total = sum(month[label] for label in MONTHLY_CATEGORY_COLUMNS)
            category_total += month[OTHER_INCLUDED_DEBITS]
            self.assertEqual(category_total, month["Monthly Spending"])

        observations = monthly_spending_observations(view)
        self.assertIn("February 2026 had your highest spending", observations[0])
        self.assertIn("January 2026 had your lowest spending", observations[1])


if __name__ == "__main__":
    unittest.main()
