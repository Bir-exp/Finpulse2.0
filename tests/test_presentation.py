import unittest

from finpulse.presentation import (
    build_overview_metrics,
    format_inr,
    format_percentage,
    largest_spending_category,
    spending_metric_label,
    spending_period_context,
)


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
            {"finpulse_score": 72, "score_band": "Good", "is_provisional": True},
        )
        self.assertEqual(overview["monthly_spending"], 30_000)
        self.assertNotEqual(overview["monthly_spending"], 90_000)
        self.assertEqual(overview["spending_label"], "Average Monthly Spending")
        self.assertIn("3 months", overview["period_context"])
        self.assertTrue(overview["provisional"])

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


if __name__ == "__main__":
    unittest.main()
