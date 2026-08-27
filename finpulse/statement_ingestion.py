"""Read and standardize CSV/XLSX bank statements.

This module deliberately has no Streamlit or database dependency.  It stops
when required columns are missing or ambiguous and returns enough structured
metadata for a later manual-mapping interface to resolve the problem.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
import re
from typing import BinaryIO, Iterable, Mapping
import unicodedata

import numpy as np
import pandas as pd


STANDARD_COLUMNS = [
    "date",
    "transaction_id",
    "debit",
    "credit",
    "balance",
    "description",
    "amount",
    "transaction_type",
]


COLUMN_ALIASES = {
    "date": {
        "date",
        "txn date",
        "transaction date",
        "value date",
        "posting date",
        "posted date",
    },
    "transaction_id": {
        "transaction id",
        "transactionid",
        "transaction number",
        "txn id",
        "reference no",
        "reference number",
        "ref no",
        "ref number",
        "transaction reference",
        "utr",
        "utr no",
        "utr number",
    },
    "debit": {
        "withdraw",
        "withdrawal",
        "withdraw amount",
        "withdrawal amount",
        "debit",
        "debit amount",
        "amount debited",
    },
    "credit": {
        "deposit",
        "deposits",
        "deposit amount",
        "credit",
        "credit amount",
        "amount credited",
    },
    "balance": {
        "balance",
        "closing balance",
        "available balance",
        "running balance",
    },
    "description": {
        "remarks",
        "remark",
        "description",
        "narration",
        "particulars",
        "transaction details",
        "details",
    },
    "amount": {
        "amount",
        "transaction amount",
        "txn amount",
    },
    "transaction_type": {
        "transaction type",
        "transactiontype",
        "txn type",
        "debit credit",
        "dr cr",
    },
}


DIRECTION_ALIASES = {
    "debit": {
        "debit",
        "dr",
        "d",
        "withdraw",
        "withdrawal",
        "amount debited",
    },
    "credit": {
        "credit",
        "cr",
        "c",
        "deposit",
        "deposits",
        "amount credited",
    },
}


EMPTY_TEXT_VALUES = {
    "",
    "-",
    "--",
    "na",
    "n a",
    "nan",
    "nat",
    "nil",
    "none",
    "null",
}


class StatementReadError(ValueError):
    """Raised when a statement file cannot be decoded or parsed."""


@dataclass(frozen=True)
class MappingResult:
    """The automatic or user-supplied source-to-schema mapping result."""

    mapping: dict[str, str]
    candidates: dict[str, tuple[str, ...]]
    ambiguous_mappings: dict[str, tuple[str, ...]]
    missing_required_fields: tuple[str, ...]
    amount_layout: str | None

    @property
    def is_valid(self) -> bool:
        return not self.missing_required_fields


@dataclass(frozen=True)
class IngestionDiagnostics:
    """Summary of mapping and row-level cleaning decisions."""

    original_row_count: int
    cleaned_row_count: int
    dropped_row_count: int
    detected_column_mapping: dict[str, str]
    missing_required_fields: tuple[str, ...]
    ambiguous_mappings: dict[str, tuple[str, ...]]
    candidate_columns: dict[str, tuple[str, ...]]
    date_range: tuple[str | None, str | None]
    debit_transactions: int
    credit_transactions: int
    dropped_row_reasons: dict[str, int]


@dataclass
class IngestionResult:
    """Standardized rows together with mapping and cleaning diagnostics."""

    transactions: pd.DataFrame
    diagnostics: IngestionDiagnostics
    mapping_result: MappingResult
    rejected_rows: pd.DataFrame

    @property
    def is_usable(self) -> bool:
        return self.mapping_result.is_valid and not self.transactions.empty


def normalize_column_name(name: object) -> str:
    """Normalize a source column name without relying on exact formatting."""

    if name is None:
        return ""

    text = unicodedata.normalize("NFKC", str(name)).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


NORMALIZED_COLUMN_ALIASES = {
    field: {normalize_column_name(alias) for alias in aliases}
    | {normalize_column_name(field)}
    for field, aliases in COLUMN_ALIASES.items()
}


NORMALIZED_DIRECTION_ALIASES = {
    direction: {normalize_column_name(alias) for alias in aliases}
    for direction, aliases in DIRECTION_ALIASES.items()
}


def _required_field_status(mapping: Mapping[str, str]) -> tuple[tuple[str, ...], str | None]:
    missing: list[str] = []

    if "date" not in mapping:
        missing.append("date")
    if "description" not in mapping:
        missing.append("description")

    has_separate_amount = "debit" in mapping or "credit" in mapping
    has_amount = "amount" in mapping
    has_direction = "transaction_type" in mapping

    if has_separate_amount:
        amount_layout = "separate_debit_credit"
    elif has_amount and has_direction:
        amount_layout = "amount_and_transaction_type"
    else:
        amount_layout = None
        if has_amount and not has_direction:
            missing.append("transaction_type")
        elif has_direction and not has_amount:
            missing.append("amount")
        else:
            missing.append("amount_and_direction")

    return tuple(missing), amount_layout


def detect_column_mapping(columns: Iterable[object]) -> MappingResult:
    """Detect standard fields and surface every ambiguous candidate."""

    source_columns = [str(column) for column in columns]
    normalized_sources = {
        source: normalize_column_name(source)
        for source in source_columns
    }

    candidates: dict[str, tuple[str, ...]] = {}
    ambiguous: dict[str, tuple[str, ...]] = {}
    mapping: dict[str, str] = {}

    for field, aliases in NORMALIZED_COLUMN_ALIASES.items():
        matches = tuple(
            source
            for source in source_columns
            if normalized_sources[source] in aliases
        )
        candidates[field] = matches

        if len(matches) == 1:
            mapping[field] = matches[0]
        elif len(matches) > 1:
            ambiguous[field] = matches

    missing, amount_layout = _required_field_status(mapping)
    return MappingResult(
        mapping=mapping,
        candidates=candidates,
        ambiguous_mappings=ambiguous,
        missing_required_fields=missing,
        amount_layout=amount_layout,
    )


def _manual_mapping_result(
    columns: Iterable[object],
    mapping: Mapping[str, str],
) -> MappingResult:
    detected = detect_column_mapping(columns)
    available_columns = {str(column) for column in columns}
    selected: dict[str, str] = {}

    for field, source in mapping.items():
        if field not in STANDARD_COLUMNS:
            raise ValueError(f"Unknown standard field in mapping: {field}")
        if str(source) not in available_columns:
            raise ValueError(f"Mapped source column does not exist: {source}")
        selected[field] = str(source)

    missing, amount_layout = _required_field_status(selected)
    unresolved_ambiguities = {
        field: sources
        for field, sources in detected.ambiguous_mappings.items()
        if field not in selected
    }
    return MappingResult(
        mapping=selected,
        candidates=detected.candidates,
        ambiguous_mappings=unresolved_ambiguities,
        missing_required_fields=missing,
        amount_layout=amount_layout,
    )


def _is_blank_value(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    if isinstance(value, str):
        return normalize_column_name(value) in EMPTY_TEXT_VALUES
    return False


def _optional_text(value: object) -> str | None:
    if _is_blank_value(value):
        return None
    return str(value).strip()


def _parse_number(value: object) -> float:
    if _is_blank_value(value):
        return float("nan")
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return float(value)

    text = unicodedata.normalize("NFKC", str(value)).strip()
    negative_parentheses = text.startswith("(") and text.endswith(")")
    text = text.replace(",", "").replace("−", "-")
    text = re.sub(r"[^0-9eE.+-]", "", text)
    if not text or text in {"+", "-", "."}:
        return float("nan")

    try:
        number = float(text)
    except ValueError:
        return float("nan")
    return -abs(number) if negative_parentheses else number


def _parse_date(value: object) -> pd.Timestamp | pd.NaT:
    if _is_blank_value(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value)
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        number = float(value)
        if 20_000 <= number <= 80_000:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(number, unit="D")

    text = str(value).strip()
    iso_style = bool(re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\D|$)", text))
    parsed = pd.to_datetime(
        text,
        errors="coerce",
        format="mixed",
        dayfirst=not iso_style,
    )
    return pd.Timestamp(parsed) if not pd.isna(parsed) else pd.NaT


def _normalize_direction(value: object) -> str | None:
    normalized = normalize_column_name(value)
    for direction, aliases in NORMALIZED_DIRECTION_ALIASES.items():
        if normalized in aliases:
            return direction
    return None


def _is_duplicate_header_row(row: pd.Series, mapping: Mapping[str, str]) -> bool:
    comparisons = []
    for source in mapping.values():
        value = row[source]
        if _is_blank_value(value):
            continue
        comparisons.append(
            normalize_column_name(value) == normalize_column_name(source)
        )
    return len(comparisons) >= 2 and sum(comparisons) / len(comparisons) >= 0.6


def _empty_transactions() -> pd.DataFrame:
    frame = pd.DataFrame(columns=STANDARD_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("debit", "credit", "balance", "amount"):
        frame[column] = pd.to_numeric(frame[column])
    return frame


def _rejected_frame(df: pd.DataFrame, rejected: list[dict[str, object]]) -> pd.DataFrame:
    if rejected:
        return pd.DataFrame(rejected)
    return pd.DataFrame(columns=[*df.columns, "_source_row", "_drop_reason"])


def _diagnostics(
    original_count: int,
    transactions: pd.DataFrame,
    mapping_result: MappingResult,
    rejected_rows: pd.DataFrame,
) -> IngestionDiagnostics:
    if transactions.empty:
        date_range: tuple[str | None, str | None] = (None, None)
        debit_count = 0
        credit_count = 0
    else:
        date_range = (
            transactions["date"].min().date().isoformat(),
            transactions["date"].max().date().isoformat(),
        )
        debit_count = int((transactions["transaction_type"] == "debit").sum())
        credit_count = int((transactions["transaction_type"] == "credit").sum())

    reason_counts = (
        Counter(rejected_rows["_drop_reason"].tolist())
        if not rejected_rows.empty
        else Counter()
    )
    return IngestionDiagnostics(
        original_row_count=original_count,
        cleaned_row_count=len(transactions),
        dropped_row_count=len(rejected_rows),
        detected_column_mapping=dict(mapping_result.mapping),
        missing_required_fields=mapping_result.missing_required_fields,
        ambiguous_mappings=dict(mapping_result.ambiguous_mappings),
        candidate_columns=dict(mapping_result.candidates),
        date_range=date_range,
        debit_transactions=debit_count,
        credit_transactions=credit_count,
        dropped_row_reasons=dict(reason_counts),
    )


def standardize_transactions(
    df: pd.DataFrame,
    mapping: Mapping[str, str] | MappingResult | None = None,
) -> IngestionResult:
    """Convert a source DataFrame into the standard transaction schema.

    A manual mapping uses standard field names as keys and source column names
    as values. Missing or ambiguous required mappings return a non-usable
    result rather than raising a row-processing exception.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if mapping is None:
        mapping_result = detect_column_mapping(df.columns)
    elif isinstance(mapping, MappingResult):
        mapping_result = mapping
    else:
        mapping_result = _manual_mapping_result(df.columns, mapping)

    original_count = len(df)
    if not mapping_result.is_valid:
        rejected = []
        for index, row in df.iterrows():
            rejected.append({
                **row.to_dict(),
                "_source_row": index,
                "_drop_reason": "schema_validation_failed",
            })
        rejected_rows = _rejected_frame(df, rejected)
        transactions = _empty_transactions()
        return IngestionResult(
            transactions=transactions,
            diagnostics=_diagnostics(
                original_count, transactions, mapping_result, rejected_rows
            ),
            mapping_result=mapping_result,
            rejected_rows=rejected_rows,
        )

    selected = mapping_result.mapping
    accepted: list[dict[str, object]] = []
    rejected = []

    for index, row in df.iterrows():
        reason: str | None = None

        if all(_is_blank_value(value) for value in row.tolist()):
            reason = "blank_row"
        elif _is_duplicate_header_row(row, selected):
            reason = "duplicate_header_row"

        parsed_date = pd.NaT
        description = None
        transaction_type = None
        debit = 0.0
        credit = 0.0
        amount = float("nan")

        if reason is None:
            parsed_date = _parse_date(row[selected["date"]])
            if pd.isna(parsed_date):
                reason = "invalid_date"

        if reason is None:
            description = _optional_text(row[selected["description"]])
            if description is None:
                reason = "missing_description"

        if reason is None and mapping_result.amount_layout == "separate_debit_credit":
            raw_debit = _parse_number(row[selected["debit"]]) if "debit" in selected else np.nan
            raw_credit = _parse_number(row[selected["credit"]]) if "credit" in selected else np.nan
            debit_value = abs(raw_debit) if not np.isnan(raw_debit) else 0.0
            credit_value = abs(raw_credit) if not np.isnan(raw_credit) else 0.0

            if debit_value > 0 and credit_value > 0:
                reason = "conflicting_debit_credit"
            elif debit_value > 0:
                debit = amount = debit_value
                transaction_type = "debit"
            elif credit_value > 0:
                credit = amount = credit_value
                transaction_type = "credit"
            else:
                reason = "missing_amount_direction"

        if reason is None and mapping_result.amount_layout == "amount_and_transaction_type":
            raw_amount = _parse_number(row[selected["amount"]])
            transaction_type = _normalize_direction(row[selected["transaction_type"]])
            if np.isnan(raw_amount) or raw_amount == 0:
                reason = "invalid_amount"
            elif transaction_type is None:
                reason = "invalid_transaction_type"
            else:
                amount = abs(raw_amount)
                if transaction_type == "debit":
                    debit = amount
                else:
                    credit = amount

        if reason is not None:
            rejected.append({
                **row.to_dict(),
                "_source_row": index,
                "_drop_reason": reason,
            })
            continue

        transaction_id = (
            _optional_text(row[selected["transaction_id"]])
            if "transaction_id" in selected
            else None
        )
        balance = (
            _parse_number(row[selected["balance"]])
            if "balance" in selected
            else np.nan
        )
        accepted.append({
            "date": parsed_date,
            "transaction_id": transaction_id,
            "debit": debit,
            "credit": credit,
            "balance": balance,
            "description": description,
            "amount": amount,
            "transaction_type": transaction_type,
        })

    transactions = pd.DataFrame(accepted, columns=STANDARD_COLUMNS)
    if transactions.empty:
        transactions = _empty_transactions()
    else:
        transactions["date"] = pd.to_datetime(transactions["date"])
        for column in ("debit", "credit", "balance", "amount"):
            transactions[column] = pd.to_numeric(transactions[column])
        transactions["transaction_id"] = transactions["transaction_id"].astype("string")

    rejected_rows = _rejected_frame(df, rejected)
    return IngestionResult(
        transactions=transactions,
        diagnostics=_diagnostics(
            original_count, transactions, mapping_result, rejected_rows
        ),
        mapping_result=mapping_result,
        rejected_rows=rejected_rows,
    )


def _source_bytes(
    file_or_path: str | Path | bytes | bytearray | BinaryIO,
    filename: str | None,
) -> tuple[bytes, str]:
    inferred_name = filename

    if isinstance(file_or_path, (str, Path)):
        path = Path(file_or_path)
        data = path.read_bytes()
        inferred_name = inferred_name or path.name
    elif isinstance(file_or_path, (bytes, bytearray)):
        data = bytes(file_or_path)
    elif hasattr(file_or_path, "read"):
        stream = file_or_path
        position = stream.tell() if hasattr(stream, "tell") else None
        raw = stream.read()
        if position is not None and hasattr(stream, "seek"):
            stream.seek(position)
        data = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        inferred_name = inferred_name or getattr(stream, "name", None)
    else:
        raise TypeError("file_or_path must be a path, bytes, or readable file object")

    return data, inferred_name or "statement.csv"


def _read_csv(data: bytes) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(
                BytesIO(data),
                sep=None,
                engine="python",
                dtype=object,
                keep_default_na=False,
                skip_blank_lines=False,
                encoding=encoding,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            errors.append(f"{encoding}: {error}")
    raise StatementReadError("Unable to parse CSV: " + "; ".join(errors))


def read_statement(
    file_or_path: str | Path | bytes | bytearray | BinaryIO,
    filename: str | None = None,
) -> IngestionResult:
    """Read a CSV/XLSX source and return standardized transactions."""

    data, source_name = _source_bytes(file_or_path, filename)
    suffix = Path(source_name).suffix.casefold()

    try:
        if suffix == ".xlsx":
            frame = pd.read_excel(
                BytesIO(data),
                engine="openpyxl",
                dtype=object,
                keep_default_na=False,
            )
        elif suffix == ".csv" or not suffix:
            frame = _read_csv(data)
        else:
            raise StatementReadError(
                f"Unsupported statement format '{suffix}'. Use CSV or XLSX."
            )
    except StatementReadError:
        raise
    except Exception as error:
        raise StatementReadError(
            f"Unable to read statement '{source_name}': {error}"
        ) from error

    return standardize_transactions(frame)
