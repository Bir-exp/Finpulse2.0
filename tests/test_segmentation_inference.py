import sqlite3
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from finpulse.segmentation_inference import (
    DEFAULT_BUNDLE_PATH,
    FEATURE_NAMES,
    load_segmentation_bundle,
    predict_persona,
    prepare_segmentation_features,
    validate_segmentation_bundle,
)
from scripts.segmentation_engine import (
    FEATURE_COLUMNS,
    SEGMENT_NAMES,
    build_segments,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "database" / "finpulse.db"


def upload_result(
    features,
    *,
    complete_months=6,
    transaction_count=60,
    analytical_confidence="Medium",
):
    values = dict(features)
    values["complete_months_observed"] = complete_months
    return SimpleNamespace(
        behavioral_features=values,
        data_quality={
            "complete_months": complete_months,
            "transaction_count": transaction_count,
            "analytical_confidence": analytical_confidence,
        },
    )


class SegmentationInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_segmentation_bundle()
        cls.center_features = {
            name: cls.bundle["cluster_centers_original_scale"][2][name]
            for name in FEATURE_NAMES
        }

    def test_frozen_artifact_exists_and_loads(self):
        self.assertTrue(DEFAULT_BUNDLE_PATH.is_file())
        self.assertEqual(
            self.bundle["artifact_type"],
            "finpulse-kmeans-segmentation-bundle",
        )

    def test_feature_order_is_explicitly_validated(self):
        altered = dict(self.bundle)
        altered["feature_names"] = tuple(reversed(FEATURE_NAMES))
        with self.assertRaisesRegex(ValueError, "feature contract mismatch"):
            validate_segmentation_bundle(altered)

    def test_existing_synthetic_users_reproduce_expected_clusters_and_personas(self):
        uri = f"file:{DATABASE_PATH.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            features = pd.read_sql_query(
                "SELECT * FROM user_features ORDER BY user_id", connection
            )
            expected = pd.read_sql_query(
                "SELECT * FROM user_segments ORDER BY user_id", connection
            )
        scaled = self.bundle["scaler"].transform(features[list(FEATURE_NAMES)])
        predicted = self.bundle["model"].predict(scaled)
        predicted_personas = pd.Series(predicted).map(
            self.bundle["cluster_persona_mapping"]
        )
        self.assertTrue(np.array_equal(predicted, expected["cluster_id"].to_numpy()))
        self.assertEqual(predicted_personas.tolist(), expected["segment_name"].tolist())

    def test_uploaded_inference_never_calls_fit_or_fit_transform(self):
        source = upload_result(self.center_features)
        with (
            mock.patch.object(StandardScaler, "fit", side_effect=AssertionError("fit")),
            mock.patch.object(
                StandardScaler,
                "fit_transform",
                side_effect=AssertionError("fit_transform"),
            ),
            mock.patch.object(KMeans, "fit", side_effect=AssertionError("fit")),
            mock.patch.object(
                KMeans, "fit_predict", side_effect=AssertionError("fit_predict")
            ),
        ):
            result = predict_persona(source, bundle=self.bundle)
        self.assertTrue(result.persona_available)

    def test_single_month_is_handled_conservatively(self):
        result = predict_persona(
            upload_result(self.center_features, complete_months=1),
            bundle=self.bundle,
        )
        self.assertFalse(result.persona_available)
        self.assertEqual(result.persona_confidence, "Unavailable")
        self.assertTrue(any("6 complete" in reason for reason in result.reasons))

    def test_missing_income_cv_is_not_zero_filled(self):
        features = dict(self.center_features)
        features["income_cv"] = None
        prepared = prepare_segmentation_features(upload_result(features))
        self.assertFalse(prepared.ready)
        self.assertIsNone(prepared.feature_values["income_cv"])
        self.assertTrue(any("income_cv" in reason for reason in prepared.reasons))

    def test_missing_volatility_is_not_zero_filled(self):
        features = dict(self.center_features)
        features["expense_ratio_volatility"] = None
        prepared = prepare_segmentation_features(upload_result(features))
        self.assertFalse(prepared.ready)
        self.assertIsNone(prepared.feature_values["expense_ratio_volatility"])

    def test_required_feature_mismatch_has_clear_diagnostics(self):
        features = dict(self.center_features)
        features.pop("avg_repayment_ratio")
        result = predict_persona(upload_result(features), bundle=self.bundle)
        self.assertFalse(result.persona_available)
        self.assertTrue(
            any("avg_repayment_ratio is unavailable" in reason for reason in result.reasons)
        )

    def test_fully_compatible_vector_produces_persona(self):
        result = predict_persona(
            upload_result(self.center_features), bundle=self.bundle
        )
        self.assertTrue(result.persona_available)
        self.assertEqual(result.cluster_id, 2)
        self.assertEqual(result.persona_name, "Consistent Savers")

    def test_persona_result_includes_model_version(self):
        result = predict_persona(
            upload_result(self.center_features), bundle=self.bundle
        )
        self.assertEqual(result.model_version, "finpulse-kmeans-v1.0.0")

    def test_cluster_persona_mapping_is_stable(self):
        self.assertEqual(self.bundle["cluster_persona_mapping"], SEGMENT_NAMES)
        self.assertEqual(
            self.bundle["reference_metadata"]["cluster_agreement_with_database"],
            1.0,
        )
        self.assertEqual(
            self.bundle["reference_metadata"]["persona_agreement_with_database"],
            1.0,
        )

    def test_extreme_out_of_distribution_user_is_withheld(self):
        extreme = {name: 1_000_000.0 for name in FEATURE_NAMES}
        result = predict_persona(upload_result(extreme), bundle=self.bundle)
        self.assertFalse(result.persona_available)
        self.assertEqual(
            result.ood_diagnostics["status"], "severe_out_of_distribution"
        )
        self.assertEqual(result.persona_confidence, "Unavailable")

    def test_typical_reference_like_user_is_not_falsely_flagged(self):
        result = predict_persona(
            upload_result(self.center_features), bundle=self.bundle
        )
        self.assertEqual(
            result.ood_diagnostics["status"], "within_reference_range"
        )
        self.assertTrue(result.persona_available)

    def test_cluster_distance_diagnostics_are_finite(self):
        result = predict_persona(
            upload_result(self.center_features), bundle=self.bundle
        )
        self.assertTrue(np.isfinite(result.distance_to_cluster_center))
        self.assertTrue(np.isfinite(result.ood_diagnostics["distance"]))
        self.assertIn("not probabilities", result.ood_diagnostics["note"])

    def test_uploaded_inference_does_not_access_sqlite(self):
        with mock.patch("sqlite3.connect", side_effect=AssertionError("database access")):
            result = predict_persona(
                upload_result(self.center_features), bundle=self.bundle
            )
        self.assertTrue(result.persona_available)

    def test_short_transaction_history_is_unavailable(self):
        result = predict_persona(
            upload_result(self.center_features, transaction_count=29),
            bundle=self.bundle,
        )
        self.assertFalse(result.persona_available)
        self.assertTrue(any("30 transactions" in reason for reason in result.reasons))

    def test_existing_v1_segmentation_code_remains_operational(self):
        uri = f"file:{DATABASE_PATH.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            features = pd.read_sql_query(
                "SELECT * FROM user_features ORDER BY user_id LIMIT 100", connection
            )
        segments, model = build_segments(features)
        self.assertEqual(tuple(FEATURE_COLUMNS), FEATURE_NAMES)
        self.assertEqual(len(segments), 100)
        self.assertEqual(model.n_clusters, 5)


if __name__ == "__main__":
    unittest.main()
