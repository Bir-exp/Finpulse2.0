"""Export the existing FinPulse v1 K-Means as a validated frozen artifact.

This is a one-time maintenance command.  It reads ``finpulse.db`` in SQLite
read-only mode, reproduces the existing v1 fit exactly, and refuses to write an
artifact unless every stored cluster and persona assignment agrees.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.segmentation_engine import (  # noqa: E402
    FEATURE_COLUMNS,
    N_CLUSTERS,
    RANDOM_STATE,
    SEGMENT_DESCRIPTIONS,
    SEGMENT_NAMES,
)


DEFAULT_DATABASE = PROJECT_ROOT / "database" / "finpulse.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "finpulse_kmeans_v1.joblib"
MODEL_VERSION = "finpulse-kmeans-v1.0.0"
ARTIFACT_TYPE = "finpulse-kmeans-segmentation-bundle"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _load_reference_tables(
    database_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uri = f"file:{database_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        features = pd.read_sql_query(
            "SELECT * FROM user_features ORDER BY user_id", connection
        )
        segments = pd.read_sql_query(
            "SELECT * FROM user_segments ORDER BY user_id", connection
        )
        profiles = pd.read_sql_query(
            "SELECT * FROM cluster_profiles ORDER BY cluster_id", connection
        )
    return features, segments, profiles


def _validate_inputs(
    features: pd.DataFrame,
    segments: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    required_features = {"user_id", *FEATURE_COLUMNS}
    required_segments = {
        "user_id",
        "cluster_id",
        "segment_name",
        "segment_description",
    }
    missing_features = sorted(required_features - set(features.columns))
    missing_segments = sorted(required_segments - set(segments.columns))
    if missing_features:
        raise RuntimeError(
            "Reference user_features is missing: " + ", ".join(missing_features)
        )
    if missing_segments:
        raise RuntimeError(
            "Reference user_segments is missing: " + ", ".join(missing_segments)
        )
    if features.empty or len(features) != len(segments):
        raise RuntimeError("Reference feature/segment row counts do not match")
    if features["user_id"].duplicated().any() or segments["user_id"].duplicated().any():
        raise RuntimeError("Reference data contains duplicate user IDs")
    if features[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Reference model features contain missing values")
    if not np.isfinite(features[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise RuntimeError("Reference model features contain non-finite values")
    if profiles.empty:
        raise RuntimeError("Reference cluster_profiles is empty")


def _mapping_from_reference(segments: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    names: dict[int, str] = {}
    descriptions: dict[int, str] = {}
    for cluster_id, rows in segments.groupby("cluster_id"):
        cluster = int(cluster_id)
        unique_names = rows["segment_name"].dropna().unique().tolist()
        unique_descriptions = rows["segment_description"].dropna().unique().tolist()
        if len(unique_names) != 1 or len(unique_descriptions) != 1:
            raise RuntimeError(
                f"Cluster {cluster} does not have exactly one stored persona mapping"
            )
        names[cluster] = str(unique_names[0])
        descriptions[cluster] = str(unique_descriptions[0])

    if names != SEGMENT_NAMES or descriptions != SEGMENT_DESCRIPTIONS:
        raise RuntimeError(
            "Stored cluster/persona mapping does not match segmentation_engine.py"
        )
    return names, descriptions


def _validate_profiles(
    features: pd.DataFrame,
    labels: np.ndarray,
    stored_profiles: pd.DataFrame,
) -> pd.DataFrame:
    calculated = features[["user_id", *FEATURE_COLUMNS]].copy()
    calculated["cluster_id"] = labels
    calculated["segment_name"] = calculated["cluster_id"].map(SEGMENT_NAMES)
    calculated = (
        calculated.groupby(["cluster_id", "segment_name"])[FEATURE_COLUMNS]
        .mean()
        .round(4)
        .reset_index()
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )
    expected = stored_profiles[
        ["cluster_id", "segment_name", *FEATURE_COLUMNS]
    ].sort_values("cluster_id").reset_index(drop=True)
    identity_matches = (
        calculated["cluster_id"].astype(int).tolist()
        == expected["cluster_id"].astype(int).tolist()
        and calculated["segment_name"].tolist()
        == expected["segment_name"].tolist()
    )
    if not identity_matches or not np.allclose(
        calculated[FEATURE_COLUMNS].to_numpy(dtype=float),
        expected[FEATURE_COLUMNS].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-5,
    ):
        raise RuntimeError("Reproduced cluster profiles do not match finpulse.db")
    return calculated


def export_bundle(
    database_path: str | Path = DEFAULT_DATABASE,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Reproduce, validate, and atomically export the current v1 model."""

    database = Path(database_path)
    output = Path(output_path)
    if not database.is_file():
        raise FileNotFoundError(f"FinPulse database was not found: {database}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing segmentation bundle: {output}"
        )

    database_hash_before = _sha256(database)
    features, stored_segments, stored_profiles = _load_reference_tables(database)
    _validate_inputs(features, stored_segments, stored_profiles)
    names, descriptions = _mapping_from_reference(stored_segments)

    matrix = features[FEATURE_COLUMNS].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    labels = model.fit_predict(scaled)

    expected_segments = features[["user_id"]].merge(
        stored_segments[["user_id", "cluster_id", "segment_name"]],
        on="user_id",
        how="left",
        validate="one_to_one",
    )
    stored_labels = expected_segments["cluster_id"].to_numpy(dtype=int)
    cluster_agreement = float(np.mean(labels == stored_labels))
    predicted_names = pd.Series(labels).map(names).to_numpy()
    stored_names = expected_segments["segment_name"].to_numpy()
    persona_agreement = float(np.mean(predicted_names == stored_names))
    if cluster_agreement != 1.0 or persona_agreement != 1.0:
        raise RuntimeError(
            "Reproduced model does not exactly match stored v1 assignments: "
            f"cluster agreement={cluster_agreement:.6f}, "
            f"persona agreement={persona_agreement:.6f}"
        )

    calculated_profiles = _validate_profiles(features, labels, stored_profiles)
    all_distances = model.transform(scaled)
    assigned_distances = all_distances[np.arange(len(labels)), labels]
    distance_by_cluster = {
        int(cluster_id): _quantiles(assigned_distances[labels == cluster_id])
        for cluster_id in range(N_CLUSTERS)
    }
    max_abs_z = np.max(np.abs(scaled), axis=1)
    centers_original = scaler.inverse_transform(model.cluster_centers_)
    feature_summary = {
        name: {
            "min": float(matrix[name].min()),
            "max": float(matrix[name].max()),
            "mean": float(matrix[name].mean()),
            "std_population": float(matrix[name].std(ddof=0)),
        }
        for name in FEATURE_COLUMNS
    }

    database_hash_after_read = _sha256(database)
    if database_hash_after_read != database_hash_before:
        raise RuntimeError("finpulse.db changed during the read-only export")

    bundle: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": tuple(FEATURE_COLUMNS),
        "scaler": scaler,
        "model": model,
        "cluster_persona_mapping": dict(names),
        "cluster_persona_descriptions": dict(descriptions),
        "cluster_centers_original_scale": {
            cluster_id: {
                name: float(centers_original[cluster_id, position])
                for position, name in enumerate(FEATURE_COLUMNS)
            }
            for cluster_id in range(N_CLUSTERS)
        },
        "reference_metadata": {
            "source": "Existing synthetic FinPulse v1 user_features in finpulse.db",
            "reference_user_count": int(len(features)),
            "database_sha256": database_hash_before,
            "cluster_counts": {
                int(cluster): int(count)
                for cluster, count in pd.Series(labels).value_counts().sort_index().items()
            },
            "cluster_agreement_with_database": cluster_agreement,
            "persona_agreement_with_database": persona_agreement,
            "assigned_distance_overall": _quantiles(assigned_distances),
            "assigned_distance_by_cluster": distance_by_cluster,
            "max_abs_standardized_feature": _quantiles(max_abs_z),
            "feature_summary": feature_summary,
            "cluster_profiles_rounded": calculated_profiles.to_dict(orient="records"),
            "minimum_upload_history": {
                "complete_calendar_months": 6,
                "transactions": 30,
                "missing_feature_policy": "withhold persona; never fill or impute",
            },
            "mapping_review_warning": (
                "The v1 labels are frozen exactly as stored. Cluster 4 has the "
                "highest income_cv while cluster 0 is named Variable-Income Users; "
                "clusters 0 and 3 also show repayment-heavy profiles. These semantic "
                "tensions are disclosed, not silently renamed in Phase 6."
            ),
        },
        "training_configuration": {
            "scaler": "sklearn.preprocessing.StandardScaler(default parameters)",
            "estimator": "sklearn.cluster.KMeans",
            "n_clusters": N_CLUSTERS,
            "random_state": RANDOM_STATE,
            "n_init": 20,
            "init": model.init,
            "algorithm": model.algorithm,
            "max_iter": model.max_iter,
            "tol": model.tol,
            "sklearn_version": sklearn.__version__,
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        joblib.dump(bundle, temporary_name)
        Path(temporary_name).replace(output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    if _sha256(database) != database_hash_before:
        output.unlink(missing_ok=True)
        raise RuntimeError("finpulse.db changed during export; artifact was removed")
    return bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact after full validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bundle = export_bundle(args.database, args.output, overwrite=args.overwrite)
    metadata = bundle["reference_metadata"]
    print(f"Exported: {args.output}")
    print(f"Model version: {bundle['model_version']}")
    print(f"Reference users: {metadata['reference_user_count']}")
    print(
        "Cluster agreement with finpulse.db: "
        f"{metadata['cluster_agreement_with_database']:.6f}"
    )
    print(
        "Persona agreement with finpulse.db: "
        f"{metadata['persona_agreement_with_database']:.6f}"
    )
    print("Cluster profiles (original feature scale):")
    for cluster_id, profile in bundle["cluster_centers_original_scale"].items():
        values = ", ".join(f"{name}={value:.6f}" for name, value in profile.items())
        print(
            f"  {cluster_id} - {bundle['cluster_persona_mapping'][cluster_id]}: {values}"
        )


if __name__ == "__main__":
    main()
