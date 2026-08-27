"""Read-only inference for the frozen FinPulse v1 behavioral segmentation.

This module never fits a model, reads SQLite, or persists uploaded data.  It
accepts the in-memory output of :mod:`finpulse.upload_analytics` and applies a
validated, versioned artifact exported from the original synthetic reference
population.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "models" / "finpulse_kmeans_v1.joblib"

FEATURE_NAMES = (
    "avg_essential_ratio",
    "avg_desire_ratio",
    "avg_repayment_ratio",
    "avg_investment_ratio",
    "avg_expense_ratio",
    "overspending_month_ratio",
    "income_cv",
    "expense_ratio_volatility",
)

# These are provenance labels, not imputation rules.  Every feature is still
# required and must be finite before inference is allowed.
FEATURE_AVAILABILITY_CLASSES = {
    "avg_essential_ratio": "A: directly computable from reviewed transactions",
    "avg_desire_ratio": "A: directly computable from reviewed transactions",
    "avg_repayment_ratio": "A: directly computable from reviewed transactions",
    "avg_investment_ratio": "A: directly computable from reviewed transactions",
    "avg_expense_ratio": "A: directly computable from reviewed transactions",
    "overspending_month_ratio": "B: requires representative monthly history",
    "income_cv": "B: requires multiple complete months with observed income",
    "expense_ratio_volatility": "B: requires multiple complete months",
}

MIN_COMPLETE_MONTHS = 6
MIN_TRANSACTIONS = 30
PERSONA_CONFIDENCE_LEVELS = ("High", "Medium", "Low", "Unavailable")


@dataclass(frozen=True)
class SegmentationFeaturePreparation:
    ready: bool
    feature_values: dict[str, float | None]
    feature_availability: dict[str, str]
    reasons: tuple[str, ...]
    complete_months: int
    transaction_count: int


@dataclass(frozen=True)
class PersonaPrediction:
    persona_available: bool
    cluster_id: int | None
    persona_name: str | None
    persona_description: str | None
    distance_to_cluster_center: float | None
    persona_confidence: str
    reasons: tuple[str, ...]
    feature_values: dict[str, float | None]
    feature_availability: dict[str, str]
    model_version: str
    ood_diagnostics: dict[str, Any]


def _field(source: object, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _as_nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        return 0
    number = float(value)
    if not np.isfinite(number) or number < 0:
        return 0
    return int(number)


def validate_segmentation_bundle(bundle: object) -> dict[str, Any]:
    """Validate the frozen artifact contract before it can be used."""

    if not isinstance(bundle, dict):
        raise ValueError("Segmentation bundle must be a dictionary")

    required = {
        "artifact_type",
        "schema_version",
        "model_version",
        "feature_names",
        "scaler",
        "model",
        "cluster_persona_mapping",
        "cluster_persona_descriptions",
        "reference_metadata",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(
            "Segmentation bundle is missing required fields: " + ", ".join(missing)
        )
    if bundle["artifact_type"] != "finpulse-kmeans-segmentation-bundle":
        raise ValueError("Unsupported segmentation artifact type")
    if tuple(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError(
            "Segmentation feature contract mismatch: expected exactly "
            + ", ".join(FEATURE_NAMES)
        )

    scaler = bundle["scaler"]
    model = bundle["model"]
    if not callable(getattr(scaler, "transform", None)):
        raise ValueError("Segmentation bundle scaler is not fitted/usable")
    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "transform", None)
    ):
        raise ValueError("Segmentation bundle model is not fitted/usable")
    if not hasattr(scaler, "mean_") or not hasattr(model, "cluster_centers_"):
        raise ValueError("Segmentation bundle contains unfitted estimators")

    mapping = bundle["cluster_persona_mapping"]
    descriptions = bundle["cluster_persona_descriptions"]
    expected_clusters = set(range(int(model.n_clusters)))
    try:
        mapping_clusters = {int(key) for key in mapping}
        description_clusters = {int(key) for key in descriptions}
    except (TypeError, ValueError) as exc:
        raise ValueError("Segmentation persona mappings have invalid cluster IDs") from exc
    if mapping_clusters != expected_clusters or description_clusters != expected_clusters:
        raise ValueError("Segmentation persona mappings do not cover every cluster")

    metadata = bundle["reference_metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Segmentation reference metadata must be a dictionary")
    distance = metadata.get("assigned_distance_by_cluster")
    max_z = metadata.get("max_abs_standardized_feature")
    if not isinstance(distance, dict) or not isinstance(max_z, dict):
        raise ValueError("Segmentation bundle lacks OOD reference distributions")
    for cluster_id in expected_clusters:
        profile = distance.get(cluster_id, distance.get(str(cluster_id)))
        if not isinstance(profile, dict) or "q75" not in profile or "q99" not in profile:
            raise ValueError(f"Missing distance reference for cluster {cluster_id}")
    if "q99" not in max_z:
        raise ValueError("Missing standardized-feature OOD reference")
    return bundle


def load_segmentation_bundle(
    path: str | Path = DEFAULT_BUNDLE_PATH,
) -> dict[str, Any]:
    """Load and validate a trusted local FinPulse segmentation artifact."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Frozen segmentation bundle was not found: {artifact_path}"
        )
    return validate_segmentation_bundle(joblib.load(artifact_path))


def prepare_segmentation_features(
    upload_analytics: object,
) -> SegmentationFeaturePreparation:
    """Prepare the exact v1 vector without filling unavailable history fields."""

    features = _field(upload_analytics, "behavioral_features", {})
    quality = _field(upload_analytics, "data_quality", {})
    if not isinstance(features, Mapping):
        features = {}
    if not isinstance(quality, Mapping):
        quality = {}

    complete_months = _as_nonnegative_int(
        quality.get("complete_months", features.get("complete_months_observed", 0))
    )
    transaction_count = _as_nonnegative_int(quality.get("transaction_count", 0))
    prepared: dict[str, float | None] = {}
    availability: dict[str, str] = {}
    reasons: list[str] = []

    for name in FEATURE_NAMES:
        value = features.get(name)
        if isinstance(value, bool) or not isinstance(value, Real):
            prepared[name] = None
            availability[name] = "Unavailable"
            reasons.append(f"{name} is unavailable; it was not imputed")
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            prepared[name] = None
            availability[name] = "Unavailable"
            reasons.append(f"{name} is non-finite; it was not imputed")
            continue
        prepared[name] = numeric
        availability[name] = FEATURE_AVAILABILITY_CLASSES[name]

    if complete_months < MIN_COMPLETE_MONTHS:
        reasons.append(
            f"At least {MIN_COMPLETE_MONTHS} complete calendar months are required "
            f"for persona inference; found {complete_months}"
        )
    if transaction_count < MIN_TRANSACTIONS:
        reasons.append(
            f"At least {MIN_TRANSACTIONS} transactions are required for persona "
            f"inference; found {transaction_count}"
        )

    return SegmentationFeaturePreparation(
        ready=not reasons,
        feature_values=prepared,
        feature_availability=availability,
        reasons=tuple(reasons),
        complete_months=complete_months,
        transaction_count=transaction_count,
    )


def _mapping_value(mapping: Mapping[Any, Any], cluster_id: int) -> Any:
    if cluster_id in mapping:
        return mapping[cluster_id]
    return mapping[str(cluster_id)]


def predict_persona(
    upload_analytics: object,
    *,
    bundle: dict[str, Any] | None = None,
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
) -> PersonaPrediction:
    """Infer a v1 persona using only frozen ``transform``/``predict`` calls."""

    frozen = validate_segmentation_bundle(bundle) if bundle is not None else load_segmentation_bundle(bundle_path)
    prepared = prepare_segmentation_features(upload_analytics)
    version = str(frozen["model_version"])
    if not prepared.ready:
        return PersonaPrediction(
            persona_available=False,
            cluster_id=None,
            persona_name=None,
            persona_description=None,
            distance_to_cluster_center=None,
            persona_confidence="Unavailable",
            reasons=prepared.reasons,
            feature_values=prepared.feature_values,
            feature_availability=prepared.feature_availability,
            model_version=version,
            ood_diagnostics={
                "evaluated": False,
                "status": "not_evaluated",
                "note": "OOD distance requires a complete eligible feature vector.",
            },
        )

    frame = pd.DataFrame(
        [[prepared.feature_values[name] for name in FEATURE_NAMES]],
        columns=list(FEATURE_NAMES),
    )
    scaled = frozen["scaler"].transform(frame)
    cluster_id = int(frozen["model"].predict(scaled)[0])
    distances = frozen["model"].transform(scaled)[0]
    distance = float(distances[cluster_id])
    max_abs_z = float(np.max(np.abs(scaled[0])))

    reference = frozen["reference_metadata"]
    distance_reference = reference["assigned_distance_by_cluster"]
    cluster_reference = distance_reference.get(
        cluster_id, distance_reference.get(str(cluster_id))
    )
    z_reference = reference["max_abs_standardized_feature"]
    q75 = float(cluster_reference["q75"])
    q99 = float(cluster_reference["q99"])
    reference_z_q99 = float(z_reference["q99"])
    severe_distance_threshold = max(q99 * 1.5, q99 + 0.5)
    severe_z_threshold = max(reference_z_q99 * 1.5, 6.0)

    severe_ood = (
        distance > severe_distance_threshold or max_abs_z > severe_z_threshold
    )
    borderline_ood = distance > q99 or max_abs_z > reference_z_q99
    reasons: list[str] = []
    if severe_ood:
        reasons.append(
            "Feature profile is far outside the synthetic v1 reference population; "
            "persona is withheld."
        )
        status = "severe_out_of_distribution"
        confidence = "Unavailable"
        available = False
    elif borderline_ood:
        reasons.append(
            "Feature profile is outside the usual synthetic v1 reference range."
        )
        status = "borderline_out_of_distribution"
        confidence = "Low"
        available = True
    else:
        status = "within_reference_range"
        quality = _field(upload_analytics, "data_quality", {})
        analytical_confidence = (
            quality.get("analytical_confidence")
            if isinstance(quality, Mapping)
            else None
        )
        confidence = (
            "High"
            if prepared.complete_months >= 12
            and analytical_confidence == "High"
            and distance <= q75
            else "Medium"
        )
        available = True

    diagnostics = {
        "evaluated": True,
        "status": status,
        "distance": distance,
        "cluster_reference_q75": q75,
        "cluster_reference_q99": q99,
        "severe_distance_threshold": severe_distance_threshold,
        "max_abs_standardized_feature": max_abs_z,
        "reference_max_abs_standardized_feature_q99": reference_z_q99,
        "severe_standardized_feature_threshold": severe_z_threshold,
        "note": (
            "Distances are diagnostics in standardized feature space, not "
            "probabilities or calibrated certainty scores."
        ),
    }
    return PersonaPrediction(
        persona_available=available,
        cluster_id=cluster_id,
        persona_name=(
            str(_mapping_value(frozen["cluster_persona_mapping"], cluster_id))
            if available
            else None
        ),
        persona_description=(
            str(
                _mapping_value(
                    frozen["cluster_persona_descriptions"], cluster_id
                )
            )
            if available
            else None
        ),
        distance_to_cluster_center=distance,
        persona_confidence=confidence,
        reasons=tuple(reasons),
        feature_values=prepared.feature_values,
        feature_availability=prepared.feature_availability,
        model_version=version,
        ood_diagnostics=diagnostics,
    )
