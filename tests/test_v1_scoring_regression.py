import unittest

import pandas as pd

from scripts.score_engine import calculate_scores


class V1ScoringRegressionTests(unittest.TestCase):
    def test_original_four_component_score_contract_is_unchanged(self):
        features = pd.DataFrame([
            {
                "user_id": "SYNTHETIC_REGRESSION",
                "avg_expense_ratio": 0.80,
                "overspending_month_ratio": 0.0,
                "avg_investment_ratio": 0.10,
                "investment_ratio_change_3m": 0.0,
                "avg_repayment_ratio": 0.10,
                "repayment_ratio_change_3m": 0.0,
                "income_cv": 0.05,
                "expense_ratio_volatility": 0.05,
                "avg_surplus_ratio": 0.20,
            }
        ])
        result = calculate_scores(features).iloc[0]
        self.assertEqual(result["spending_control_score"], 27)
        self.assertEqual(result["savings_score"], 18)
        self.assertEqual(result["debt_management_score"], 23)
        self.assertEqual(result["stability_score"], 20)
        self.assertEqual(result["finpulse_score"], 88)
        self.assertEqual(result["score_band"], "Strong")


if __name__ == "__main__":
    unittest.main()
