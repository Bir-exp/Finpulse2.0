"""Reusable helpers for reviewing and correcting predicted categories."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import pandas as pd


FINPULSE_CATEGORIES = (
    "Income",
    "Essentials",
    "Desire",
    "Repayment",
    "Investment/Savings",
    "Others",
)
ALLOWED_CATEGORY_SET = frozenset(FINPULSE_CATEGORIES)


def _require_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame or None")
    return df


def _invalid_categories(values: pd.Series) -> list[object]:
    invalid: list[object] = []
    seen: set[str] = set()
    for value in values.tolist():
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        try:
            allowed = value in ALLOWED_CATEGORY_SET
        except TypeError:
            allowed = False
        if missing or not allowed:
            identity = "<missing>" if missing else f"{type(value).__name__}:{value!r}"
            if identity not in seen:
                invalid.append(value)
                seen.add(identity)
    return invalid


def initialize_review_categories(
    df: pd.DataFrame | None,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Initialize category review and behavioral-analysis inclusion fields.

    Existing valid final categories are preserved unless ``overwrite`` is
    explicitly requested. Existing session frames without
    ``include_in_analysis`` safely default to ``True``.
    """

    source = _require_dataframe(df)
    result = source.copy()

    if result.empty:
        if "final_category" not in result.columns:
            result["final_category"] = pd.Series(dtype="string", index=result.index)
        if "include_in_analysis" not in result.columns:
            result["include_in_analysis"] = pd.Series(
                dtype="bool", index=result.index
            )
        return result

    if "predicted_category" not in result.columns:
        raise ValueError("predicted_category is required to initialize review categories")

    invalid_predictions = _invalid_categories(result["predicted_category"])
    if invalid_predictions:
        raise ValueError(
            "predicted_category contains unsupported values: "
            + ", ".join(repr(value) for value in invalid_predictions)
        )

    if "final_category" in result.columns and not overwrite:
        validate_final_categories(result)
    else:
        result["final_category"] = result["predicted_category"].astype("string")

    if overwrite or "include_in_analysis" not in result.columns:
        result["include_in_analysis"] = True
    validate_include_in_analysis(result)
    return result


def validate_final_categories(df: pd.DataFrame | None) -> bool:
    """Validate that every non-empty row has one allowed final category."""

    transactions = _require_dataframe(df)
    if transactions.empty:
        return True
    if "final_category" not in transactions.columns:
        raise ValueError("final_category is required before review confirmation")

    invalid = _invalid_categories(transactions["final_category"])
    if invalid:
        allowed = ", ".join(FINPULSE_CATEGORIES)
        invalid_text = ", ".join(repr(value) for value in invalid)
        raise ValueError(
            f"Invalid final_category value(s): {invalid_text}. Allowed values: {allowed}"
        )
    return True


def validate_include_in_analysis(df: pd.DataFrame | None) -> bool:
    """Require one explicit boolean inclusion decision for every row."""

    transactions = _require_dataframe(df)
    if transactions.empty:
        return True
    if "include_in_analysis" not in transactions.columns:
        raise ValueError("include_in_analysis is required before review confirmation")
    invalid = [
        value
        for value in transactions["include_in_analysis"].tolist()
        if not isinstance(value, (bool, np.bool_))
    ]
    if invalid:
        raise ValueError(
            "include_in_analysis must contain only True or False values"
        )
    return True


def validate_review_fields(df: pd.DataFrame | None) -> bool:
    """Validate both editable review decisions."""

    validate_final_categories(df)
    validate_include_in_analysis(df)
    return True


def apply_review_edits(
    df: pd.DataFrame | None,
    edits: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge filtered editor changes without dropping or mutating predictions."""

    result = initialize_review_categories(df)
    if edits is None or len(edits) == 0:
        return result
    if not isinstance(edits, pd.DataFrame):
        raise TypeError("review edits must be a pandas DataFrame or None")
    if not result.index.is_unique or not edits.index.is_unique:
        raise ValueError("Transaction and edit indices must be unique")

    editable = [
        column
        for column in ("final_category", "include_in_analysis")
        if column in edits.columns
    ]
    if not editable:
        raise ValueError(
            "Edited DataFrame must contain final_category or include_in_analysis"
        )
    unknown_indices = [index for index in edits.index if index not in result.index]
    if unknown_indices:
        raise KeyError(f"Edit indices are not present in transactions: {unknown_indices}")

    if "final_category" in editable:
        invalid = _invalid_categories(edits["final_category"])
        if invalid:
            allowed = ", ".join(FINPULSE_CATEGORIES)
            invalid_text = ", ".join(repr(value) for value in invalid)
            raise ValueError(
                f"Invalid manual category value(s): {invalid_text}. "
                f"Allowed values: {allowed}"
            )
    if "include_in_analysis" in editable:
        invalid_inclusion = [
            value
            for value in edits["include_in_analysis"].tolist()
            if not isinstance(value, (bool, np.bool_))
        ]
        if invalid_inclusion:
            raise ValueError(
                "include_in_analysis edits must contain only True or False"
            )

    original_predictions = result["predicted_category"].copy()
    original_row_count = len(result)
    for column in editable:
        for index, value in edits[column].items():
            result.at[index, column] = bool(value) if column == "include_in_analysis" else value

    validate_review_fields(result)
    if len(result) != original_row_count:
        raise RuntimeError("Review edits unexpectedly changed transaction row count")
    if not result["predicted_category"].equals(original_predictions):
        raise RuntimeError("Review edits must not change predicted_category")
    return result


def apply_category_edits(
    df: pd.DataFrame | None,
    edits: Mapping[Hashable, str] | pd.DataFrame | None,
) -> pd.DataFrame:
    """Apply index-addressed category edits without altering predictions.

    ``edits`` may be a mapping of DataFrame index to category, or an edited
    DataFrame/subset containing ``final_category``. The source index must be
    unique so filtered review tables can be merged without changing row order.
    """

    result = initialize_review_categories(df)
    if edits is None or (hasattr(edits, "__len__") and len(edits) == 0):
        return result
    if not result.index.is_unique:
        raise ValueError("Transaction index must be unique before applying edits")

    if isinstance(edits, pd.DataFrame):
        if "final_category" not in edits.columns:
            raise ValueError("Edited DataFrame must contain final_category")
        return apply_review_edits(result, edits[["final_category"]])
    elif isinstance(edits, Mapping):
        edit_map = edits
    else:
        raise TypeError("edits must be a mapping, DataFrame, or None")

    unknown_indices = [index for index in edit_map if index not in result.index]
    if unknown_indices:
        raise KeyError(f"Edit indices are not present in transactions: {unknown_indices}")

    edit_values = pd.Series(list(edit_map.values()), dtype="object")
    invalid = _invalid_categories(edit_values)
    if invalid:
        allowed = ", ".join(FINPULSE_CATEGORIES)
        invalid_text = ", ".join(repr(value) for value in invalid)
        raise ValueError(
            f"Invalid manual category value(s): {invalid_text}. Allowed values: {allowed}"
        )

    original_predictions = result["predicted_category"].copy()
    original_row_count = len(result)
    for index, category in edit_map.items():
        result.at[index, "final_category"] = category

    validate_final_categories(result)
    if len(result) != original_row_count:
        raise RuntimeError("Category edits unexpectedly changed transaction row count")
    if not result["predicted_category"].equals(original_predictions):
        raise RuntimeError("Category edits must not change predicted_category")
    return result


def filter_review_transactions(
    df: pd.DataFrame | None,
    *,
    low_confidence_only: bool = False,
) -> pd.DataFrame:
    """Return all transactions or only rows whose confidence is ``Low``."""

    transactions = _require_dataframe(df)
    if not low_confidence_only:
        return transactions.copy()
    if transactions.empty:
        return transactions.copy()
    if "confidence" not in transactions.columns:
        raise ValueError("confidence is required for low-confidence filtering")
    return transactions.loc[transactions["confidence"] == "Low"].copy()
