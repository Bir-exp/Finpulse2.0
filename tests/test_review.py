import unittest

import pandas as pd

from finpulse.review import (
    FINPULSE_CATEGORIES,
    apply_category_edits,
    apply_review_edits,
    filter_review_transactions,
    initialize_review_categories,
    validate_final_categories,
    validate_include_in_analysis,
)


def sample_transactions():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "transaction_id": ["T1", "T2", "T3"],
        "debit": [250.0, 0.0, 500.0],
        "credit": [0.0, 5000.0, 0.0],
        "balance": [750.0, 5750.0, 5250.0],
        "description": ["SWIGGY", "SALARY", "UNKNOWN SHOP"],
        "amount": [250.0, 5000.0, 500.0],
        "transaction_type": ["debit", "credit", "debit"],
        "receiver": ["Swiggy", "Salary", "Unknown Shop"],
        "detailed_category": ["Food Delivery", "Salary", "Unclassified Merchant"],
        "predicted_category": ["Desire", "Income", "Others"],
        "confidence": ["High", "High", "Low"],
    })


class ReviewTests(unittest.TestCase):
    def test_new_rows_default_to_included_in_analysis(self):
        result = initialize_review_categories(sample_transactions())
        self.assertTrue(result["include_in_analysis"].all())
        self.assertTrue(validate_include_in_analysis(result))

    def test_excluding_transaction_preserves_row(self):
        source = initialize_review_categories(sample_transactions())
        edits = source.loc[[0], ["final_category", "include_in_analysis"]].copy()
        edits.loc[0, "include_in_analysis"] = False
        result = apply_review_edits(source, edits)
        self.assertEqual(len(result), len(source))
        self.assertFalse(bool(result.loc[0, "include_in_analysis"]))
        self.assertEqual(result.loc[0, "description"], "SWIGGY")

    def test_category_and_inclusion_edits_are_independent(self):
        source = initialize_review_categories(sample_transactions())
        edits = source.loc[[0], ["final_category", "include_in_analysis"]].copy()
        edits.loc[0, "final_category"] = "Essentials"
        edits.loc[0, "include_in_analysis"] = False
        result = apply_review_edits(source, edits)
        self.assertEqual(result.loc[0, "final_category"], "Essentials")
        self.assertFalse(bool(result.loc[0, "include_in_analysis"]))

    def test_review_edits_preserve_predicted_category(self):
        source = initialize_review_categories(sample_transactions())
        predictions = source["predicted_category"].copy()
        edits = source.loc[[0], ["final_category", "include_in_analysis"]].copy()
        edits.loc[0, "final_category"] = "Essentials"
        edits.loc[0, "include_in_analysis"] = False
        result = apply_review_edits(source, edits)
        self.assertTrue(result["predicted_category"].equals(predictions))

    def test_legacy_review_frame_defaults_inclusion_to_true(self):
        legacy = initialize_review_categories(sample_transactions()).drop(
            columns="include_in_analysis"
        )
        result = initialize_review_categories(legacy)
        self.assertTrue(result["include_in_analysis"].all())

    def test_final_category_initializes_from_prediction(self):
        result = initialize_review_categories(sample_transactions())
        self.assertEqual(
            result["final_category"].tolist(),
            result["predicted_category"].tolist(),
        )

    def test_prediction_remains_unchanged_after_edit(self):
        original = initialize_review_categories(sample_transactions())
        predictions = original["predicted_category"].copy()
        result = apply_category_edits(original, {0: "Essentials"})
        self.assertTrue(result["predicted_category"].equals(predictions))
        self.assertEqual(result.loc[0, "final_category"], "Essentials")

    def test_all_six_categories_are_accepted(self):
        source = sample_transactions().iloc[[0]].copy()
        for category in FINPULSE_CATEGORIES:
            with self.subTest(category=category):
                result = apply_category_edits(
                    initialize_review_categories(source),
                    {source.index[0]: category},
                )
                self.assertTrue(validate_final_categories(result))

    def test_invalid_manual_category_fails_clearly(self):
        source = initialize_review_categories(sample_transactions())
        with self.assertRaisesRegex(ValueError, "Invalid manual category"):
            apply_category_edits(source, {0: "Dining"})

    def test_row_count_remains_unchanged(self):
        source = initialize_review_categories(sample_transactions())
        result = apply_category_edits(source, {0: "Essentials", 2: "Desire"})
        self.assertEqual(len(result), len(source))

    def test_low_confidence_filtering(self):
        source = initialize_review_categories(sample_transactions())
        result = filter_review_transactions(source, low_confidence_only=True)
        self.assertEqual(result.index.tolist(), [2])
        self.assertTrue((result["confidence"] == "Low").all())

    def test_no_edit_preserves_prediction_as_final(self):
        source = sample_transactions()
        result = apply_category_edits(source, {})
        self.assertEqual(
            result["final_category"].tolist(),
            source["predicted_category"].tolist(),
        )

    def test_multiple_corrections_apply_correctly(self):
        source = initialize_review_categories(sample_transactions())
        result = apply_category_edits(
            source,
            {0: "Essentials", 1: "Others", 2: "Repayment"},
        )
        self.assertEqual(
            result["final_category"].tolist(),
            ["Essentials", "Others", "Repayment"],
        )

    def test_none_and_empty_dataframe_are_safe(self):
        none_result = initialize_review_categories(None)
        empty_result = initialize_review_categories(pd.DataFrame())
        self.assertTrue(none_result.empty)
        self.assertTrue(empty_result.empty)
        self.assertIn("final_category", none_result.columns)
        self.assertTrue(validate_final_categories(None))
        self.assertTrue(filter_review_transactions(None, low_confidence_only=True).empty)

    def test_original_non_review_fields_are_preserved(self):
        source = initialize_review_categories(sample_transactions())
        original_columns = [column for column in source.columns if column != "final_category"]
        result = apply_category_edits(source, {2: "Essentials"})
        pd.testing.assert_frame_equal(result[original_columns], source[original_columns])

    def test_filtered_dataframe_edits_merge_by_index(self):
        source = initialize_review_categories(sample_transactions())
        low_rows = filter_review_transactions(source, low_confidence_only=True)
        low_rows.loc[2, "final_category"] = "Essentials"
        result = apply_category_edits(source, low_rows)
        self.assertEqual(result.loc[2, "final_category"], "Essentials")
        self.assertEqual(result.loc[0, "final_category"], "Desire")

    def test_existing_edits_survive_reinitialization(self):
        source = initialize_review_categories(sample_transactions())
        edited = apply_category_edits(source, {2: "Essentials"})
        rerun = initialize_review_categories(edited)
        self.assertEqual(rerun.loc[2, "final_category"], "Essentials")

    def test_invalid_final_category_validation_fails(self):
        source = initialize_review_categories(sample_transactions())
        source.loc[0, "final_category"] = "Invalid"
        with self.assertRaisesRegex(ValueError, "Invalid final_category"):
            validate_final_categories(source)

    def test_multiple_null_final_categories_fail_clearly(self):
        source = initialize_review_categories(sample_transactions())
        source.loc[[0, 1], "final_category"] = pd.NA
        with self.assertRaisesRegex(ValueError, "Invalid final_category"):
            validate_final_categories(source)


if __name__ == "__main__":
    unittest.main()
