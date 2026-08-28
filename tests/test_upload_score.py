import unittest

from finpulse.upload_score import calculate_upload_behavioral_score
from scripts.score_engine import score_band


def score_features(**overrides):
    features = {
        "avg_expense_ratio": 0.20,
        "avg_desire_ratio": 0.00,
        "overspending_month_ratio": 0.00,
        "avg_investment_ratio": 0.20,
        "avg_remaining_ratio": 0.80,
        "avg_repayment_ratio": 0.00,
    }
    features.update(overrides)
    return features


class UploadBehavioralScoreTests(unittest.TestCase):
    def test_maximum_is_a_genuine_one_hundred_points(self):
        result = calculate_upload_behavioral_score(score_features())
        self.assertEqual(result["total_score"], 100)
        self.assertEqual(result["finpulse_score"], 100)
        self.assertEqual(
            sum(component["max_score"] for component in result["components"].values()),
            100,
        )
        self.assertEqual(
            {key: component["max_score"] for key, component in result["components"].items()},
            {
                "spending_control": 40,
                "saving_investment": 35,
                "repayment_management": 25,
            },
        )

    def test_score_is_bounded_and_has_no_stability_or_renormalization(self):
        low = calculate_upload_behavioral_score(
            score_features(
                avg_expense_ratio=4.0,
                avg_desire_ratio=2.0,
                overspending_month_ratio=1.0,
                avg_investment_ratio=0.0,
                avg_remaining_ratio=-3.0,
                avg_repayment_ratio=2.0,
            )
        )
        self.assertGreaterEqual(low["total_score"], 0)
        self.assertLessEqual(low["total_score"], 100)
        self.assertNotIn("stability", low["components"])
        self.assertNotIn("available_maximum", low)
        self.assertNotIn("raw_available_score", low)
        self.assertNotIn("renormalized_for_unavailable_components", low)
        self.assertNotIn("is_provisional", low)

    def test_higher_desire_ratio_lowers_only_relevant_spending_subscore(self):
        low_desire = calculate_upload_behavioral_score(score_features())
        high_desire = calculate_upload_behavioral_score(
            score_features(avg_desire_ratio=0.30)
        )
        low_component = low_desire["components"]["spending_control"]
        high_component = high_desire["components"]["spending_control"]
        self.assertGreater(low_component["score"], high_component["score"])
        self.assertEqual(
            low_component["subcomponents"]["overall_spending"]["score"],
            high_component["subcomponents"]["overall_spending"]["score"],
        )

    def test_higher_total_spending_lowers_spending_control(self):
        controlled = calculate_upload_behavioral_score(score_features())
        pressured = calculate_upload_behavioral_score(
            score_features(avg_expense_ratio=1.20, avg_remaining_ratio=-0.20)
        )
        self.assertGreater(
            controlled["components"]["spending_control"]["score"],
            pressured["components"]["spending_control"]["score"],
        )

    def test_explicit_investment_improves_saving_component(self):
        none = calculate_upload_behavioral_score(
            score_features(avg_investment_ratio=0.0)
        )
        explicit = calculate_upload_behavioral_score(score_features())
        self.assertGreater(
            explicit["components"]["saving_investment"]["score"],
            none["components"]["saving_investment"]["score"],
        )

    def test_positive_remainder_scores_above_negative_remainder(self):
        positive = calculate_upload_behavioral_score(score_features())
        negative = calculate_upload_behavioral_score(
            score_features(avg_remaining_ratio=-0.01)
        )
        self.assertGreater(
            positive["components"]["saving_investment"]["score"],
            negative["components"]["saving_investment"]["score"],
        )

    def test_overspending_frequency_lowers_discipline(self):
        never = calculate_upload_behavioral_score(score_features())
        frequent = calculate_upload_behavioral_score(
            score_features(overspending_month_ratio=0.60)
        )
        never_discipline = never["components"]["spending_control"]["subcomponents"][
            "budget_and_overspending_discipline"
        ]
        frequent_discipline = frequent["components"]["spending_control"][
            "subcomponents"
        ]["budget_and_overspending_discipline"]
        self.assertEqual(never_discipline["score"], 8)
        self.assertEqual(frequent_discipline["score"], 0)

    def test_higher_repayment_burden_lowers_repayment_management(self):
        no_repayment = calculate_upload_behavioral_score(score_features())
        high_repayment = calculate_upload_behavioral_score(
            score_features(avg_repayment_ratio=0.45)
        )
        self.assertEqual(
            no_repayment["components"]["repayment_management"]["score"], 25
        )
        self.assertEqual(
            high_repayment["components"]["repayment_management"]["score"], 3
        )

    def test_budget_none_does_not_reduce_maximum(self):
        result = calculate_upload_behavioral_score(score_features(), None)
        discipline = result["components"]["spending_control"]["subcomponents"][
            "budget_and_overspending_discipline"
        ]
        self.assertEqual(result["total_score"], 100)
        self.assertEqual(discipline["score"], 8)
        self.assertIsNone(discipline["budget_score"])

    def test_budget_refines_only_spending_control(self):
        without_budget = calculate_upload_behavioral_score(score_features())
        over_budget = calculate_upload_behavioral_score(
            score_features(), {"budget_utilization": 1.20}
        )
        self.assertLess(
            over_budget["components"]["spending_control"]["score"],
            without_budget["components"]["spending_control"]["score"],
        )
        for component in ("saving_investment", "repayment_management"):
            self.assertEqual(
                without_budget["components"][component],
                over_budget["components"][component],
            )

    def test_existing_score_band_boundaries_are_reused(self):
        expectations = {
            100: "Strong",
            80: "Strong",
            79: "Stable",
            65: "Stable",
            64: "Watchful",
            50: "Watchful",
            49: "Strained",
            35: "Strained",
            34: "High Pressure",
            0: "High Pressure",
        }
        for score, band in expectations.items():
            with self.subTest(score=score):
                self.assertEqual(score_band(score), band)
        self.assertEqual(
            calculate_upload_behavioral_score(score_features())["score_band"],
            "Strong",
        )


if __name__ == "__main__":
    unittest.main()
