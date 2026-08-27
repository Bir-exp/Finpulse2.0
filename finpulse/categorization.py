"""Deterministic transaction categorization for standardized statements."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Mapping
import unicodedata

import pandas as pd
from rapidfuzz import fuzz

from .statement_ingestion import normalize_column_name


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RULES_FILE = PROJECT_ROOT / "config" / "category_rules.json"

ALLOWED_CATEGORIES = frozenset({
    "Income",
    "Essentials",
    "Desire",
    "Repayment",
    "Investment/Savings",
    "Others",
})
ALLOWED_CONFIDENCE = frozenset({"High", "Medium", "Low"})

CHANNEL_PREFIXES = re.compile(
    r"^(?:upi|neft|imps|rtgs|ach\s*[dc]?|ecs|pos|atm)\b[\s:.-]*",
    flags=re.IGNORECASE,
)
PAYMENT_HANDLE = re.compile(
    r"(?<!\w)[a-z0-9.*_]{2,}@[a-z0-9._-]+",
    flags=re.IGNORECASE,
)
REFERENCE_TOKEN = re.compile(
    r"\b(?:ref|reference|txn|transaction|utr)\s*(?:no|number)?\s*[a-z0-9-]{6,}\b",
    flags=re.IGNORECASE,
)
LONG_NUMBER = re.compile(r"\b\d{6,}\b")
MASKED_FRAGMENT = re.compile(r"\b(?:x{2,}|\*{2,})[a-z0-9*]*\b", flags=re.IGNORECASE)

BANK_CHANNEL_NOISE = {
    "upi",
    "neft",
    "imps",
    "rtgs",
    "ach",
    "ach d",
    "ach c",
    "ecs",
    "pos",
    "atm",
    "icici",
    "icici bank",
    "axis",
    "axis bank",
    "hdfc",
    "sbi",
    "state bank of india",
    "ybl",
    "paytm",
}

DIRECTION_VALUES = {
    "debit": "debit",
    "dr": "debit",
    "withdraw": "debit",
    "withdrawal": "debit",
    "credit": "credit",
    "cr": "credit",
    "deposit": "credit",
    "deposits": "credit",
}


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def clean_description(text: object) -> str:
    """Clean spacing/control characters while retaining narration content."""

    if _is_empty(text):
        return ""
    cleaned = unicodedata.normalize("NFKC", str(text))
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" \t\r\n|/-")


def _humanize_receiver(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .,/|-_")
    if text and (text.isupper() or text.islower()):
        return text.title()
    return text


def extract_receiver(description: object) -> str:
    """Extract a conservative human-readable merchant or person candidate."""

    cleaned = clean_description(description)
    if not cleaned:
        return ""

    without_handles = PAYMENT_HANDLE.sub(" ", cleaned)
    without_handles = REFERENCE_TOKEN.sub(" ", without_handles)
    without_handles = MASKED_FRAGMENT.sub(" ", without_handles)

    parts = re.split(r"[/|]+|\s+-\s+|-(?=[A-Za-z])", without_handles)
    candidates: list[str] = []

    for part in parts:
        candidate = LONG_NUMBER.sub(" ", part)
        candidate = CHANNEL_PREFIXES.sub("", candidate)
        candidate = re.sub(r"^(?:to|from|by)\b[\s:.-]*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .,/|-_")
        normalized = normalize_column_name(candidate)

        if not candidate or normalized in BANK_CHANNEL_NOISE:
            continue
        if normalized.isdigit():
            continue
        if candidates and normalize_column_name(candidates[-1]) == normalized:
            continue
        candidates.append(candidate)

    if not candidates:
        fallback = CHANNEL_PREFIXES.sub("", without_handles)
        fallback = LONG_NUMBER.sub(" ", fallback)
        return _humanize_receiver(fallback or cleaned)

    return _humanize_receiver(" ".join(candidates))


def _validate_rule(rule: Mapping[str, Any], allowed_categories: set[str]) -> None:
    category = rule.get("finpulse_category")
    confidence = rule.get("confidence")
    if category not in allowed_categories:
        raise ValueError(f"Invalid FinPulse category in rules: {category}")
    if confidence not in ALLOWED_CONFIDENCE:
        raise ValueError(f"Invalid confidence in rules: {confidence}")
    if not rule.get("detailed_category"):
        raise ValueError("Every category rule needs a detailed_category")


@lru_cache(maxsize=None)
def load_category_rules(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate editable categorization rules."""

    rules_path = Path(path) if path is not None else DEFAULT_RULES_FILE
    with rules_path.open("r", encoding="utf-8") as file:
        rules = json.load(file)

    configured_categories = set(rules.get("allowed_categories", []))
    if configured_categories != set(ALLOWED_CATEGORIES):
        raise ValueError("Configured allowed_categories do not match FinPulse categories")

    for section in (
        "context_rules",
        "exact_merchants",
        "strong_phrases",
        "generic_keywords",
    ):
        if not isinstance(rules.get(section), list):
            raise ValueError(f"Rule section must be a list: {section}")
        for rule in rules[section]:
            _validate_rule(rule, configured_categories)

    threshold = rules.get("fuzzy_threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 100:
        raise ValueError("fuzzy_threshold must be between 0 and 100")
    return rules


def _normalise_match_text(text: object) -> str:
    return normalize_column_name(clean_description(text))


def _contains_phrase(text: str, phrase: object) -> bool:
    normalized_phrase = _normalise_match_text(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in f" {text} "


def _direction(transaction_type: object) -> str | None:
    return DIRECTION_VALUES.get(_normalise_match_text(transaction_type))


def _direction_allows(rule: Mapping[str, Any], direction: str | None) -> bool:
    permitted = rule.get("transaction_types")
    return not permitted or direction in permitted


def _rule_result(rule: Mapping[str, Any], receiver: str) -> dict[str, str]:
    return {
        "receiver": receiver,
        "detailed_category": str(rule["detailed_category"]),
        "predicted_category": str(rule["finpulse_category"]),
        "confidence": str(rule["confidence"]),
    }


def _first_phrase_rule(
    rules: list[dict[str, Any]],
    field_name: str,
    text: str,
    direction: str | None,
) -> dict[str, Any] | None:
    for rule in rules:
        if not _direction_allows(rule, direction):
            continue
        if any(_contains_phrase(text, phrase) for phrase in rule[field_name]):
            if rule["finpulse_category"] == "Income" and direction != "credit":
                continue
            return rule
    return None


def _merchant_rule(
    rules: list[dict[str, Any]],
    description_text: str,
    receiver_text: str,
    direction: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    ambiguous_match = None
    for rule in rules:
        if not _direction_allows(rule, direction):
            continue
        aliases = rule.get("aliases", [])
        matched = any(
            _contains_phrase(receiver_text, alias)
            or _contains_phrase(description_text, alias)
            for alias in aliases
        )
        if not matched:
            continue
        if rule.get("ambiguous"):
            ambiguous_match = ambiguous_match or rule
            continue
        return rule, ambiguous_match
    return None, ambiguous_match


def _fuzzy_rule(
    rules: list[dict[str, Any]],
    receiver_text: str,
    direction: str | None,
    threshold: float,
) -> dict[str, Any] | None:
    candidate = re.sub(
        r"\b(?:limited|ltd|private|pvt|payment|paid|order)\b",
        " ",
        receiver_text,
    )
    candidate = " ".join(candidate.split())
    if len(candidate) < 5:
        return None

    best_rule = None
    best_score = 0.0
    for rule in rules:
        if rule.get("ambiguous") or not _direction_allows(rule, direction):
            continue
        for alias in rule.get("aliases", []):
            normalized_alias = _normalise_match_text(alias)
            if len(normalized_alias) < 5 or abs(len(candidate) - len(normalized_alias)) > 4:
                continue
            score = fuzz.ratio(candidate, normalized_alias)
            if score >= threshold and score > best_score:
                best_rule = rule
                best_score = score
    return best_rule


def categorize_transaction(
    description: object,
    transaction_type: object,
    receiver: object | None = None,
    rules: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Categorize one standardized transaction using deterministic evidence."""

    configured_rules = dict(rules) if rules is not None else load_category_rules()
    cleaned_description = clean_description(description)
    extracted_receiver = (
        clean_description(receiver)
        if not _is_empty(receiver)
        else extract_receiver(cleaned_description)
    )
    description_text = _normalise_match_text(cleaned_description)
    receiver_text = _normalise_match_text(extracted_receiver)
    combined_text = " ".join(part for part in (description_text, receiver_text) if part)
    direction = _direction(transaction_type)

    context = _first_phrase_rule(
        configured_rules["context_rules"],
        "phrases",
        combined_text,
        direction,
    )
    if context is not None:
        return _rule_result(context, extracted_receiver)

    merchant, ambiguous_merchant = _merchant_rule(
        configured_rules["exact_merchants"],
        description_text,
        receiver_text,
        direction,
    )
    if merchant is not None:
        return _rule_result(merchant, extracted_receiver)

    strong_phrase = _first_phrase_rule(
        configured_rules["strong_phrases"],
        "phrases",
        combined_text,
        direction,
    )
    if strong_phrase is not None:
        return _rule_result(strong_phrase, extracted_receiver)

    keyword = _first_phrase_rule(
        configured_rules["generic_keywords"],
        "keywords",
        combined_text,
        direction,
    )
    if keyword is not None:
        return _rule_result(keyword, extracted_receiver)

    fuzzy_match = _fuzzy_rule(
        configured_rules["exact_merchants"],
        receiver_text,
        direction,
        float(configured_rules["fuzzy_threshold"]),
    )
    if fuzzy_match is not None:
        fuzzy_result = _rule_result(fuzzy_match, extracted_receiver)
        fuzzy_result["confidence"] = "Medium"
        return fuzzy_result

    if ambiguous_merchant is not None:
        return _rule_result(ambiguous_merchant, extracted_receiver)

    transfer_words = ("upi", "neft", "imps", "transfer", "received from", "paid to")
    detailed = (
        "Unknown Transfer"
        if any(_contains_phrase(description_text, word) for word in transfer_words)
        else "Unclassified Merchant"
    )
    return {
        "receiver": extracted_receiver or cleaned_description,
        "detailed_category": detailed,
        "predicted_category": "Others",
        "confidence": "Low",
    }


def categorize_transactions(
    df: pd.DataFrame,
    rules: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Append Phase 3 categorization fields without changing row count."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = {"description", "transaction_type"} - set(df.columns)
    if missing:
        raise ValueError(
            "Standardized transactions are missing columns: "
            + ", ".join(sorted(missing))
        )

    result = df.copy()
    categorized = [
        categorize_transaction(
            row["description"],
            row["transaction_type"],
            rules=rules,
        )
        for _, row in result.iterrows()
    ]
    category_frame = pd.DataFrame(
        categorized,
        index=result.index,
        columns=[
            "receiver",
            "detailed_category",
            "predicted_category",
            "confidence",
        ],
    )
    for column in category_frame.columns:
        result[column] = category_frame[column]

    if not set(result["predicted_category"]).issubset(ALLOWED_CATEGORIES):
        raise ValueError("Categorization produced an unsupported FinPulse category")
    return result
