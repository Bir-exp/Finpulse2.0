from io import BytesIO
import unittest

import pandas as pd

from finpulse.statement_ingestion import (
    STANDARD_COLUMNS,
    detect_column_mapping,
    normalize_column_name,
    read_statement,
    standardize_transactions,
)


class StatementIngestionTests(unittest.TestCase):
    def test_exact_real_statement_format(self):
        frame = pd.DataFrame({
            "Date": ["01/01/2024", "02/01/2024"],
            "transactionId": ["T1", "T2"],
            "withdraw": ["1,250.50", ""],
            "deposits": ["", "10,000"],
            "balance": ["8,749.50", "18,749.50"],
            "remarks": ["Grocery store", "Salary"],
        })

        result = standardize_transactions(frame)

        self.assertTrue(result.is_usable)
        self.assertEqual(list(result.transactions.columns), STANDARD_COLUMNS)
        self.assertEqual(result.transactions["transaction_type"].tolist(), ["debit", "credit"])
        self.assertEqual(result.transactions["amount"].tolist(), [1250.50, 10000.0])
        self.assertEqual(result.diagnostics.debit_transactions, 1)
        self.assertEqual(result.diagnostics.credit_transactions, 1)

    def test_capitalization_variants(self):
        frame = pd.DataFrame({
            "DATE": ["2024-01-01"],
            "TRANSACTION ID": ["T1"],
            "WITHDRAW": [100],
            "DEPOSITS": [""],
            "BALANCE": [900],
            "REMARKS": ["Test debit"],
        })

        result = standardize_transactions(frame)
        self.assertTrue(result.is_usable)
        self.assertEqual(result.transactions.loc[0, "transaction_id"], "T1")

    def test_underscore_hyphen_and_camel_case_variants(self):
        columns = [
            "txn_date",
            "TransactionID",
            "withdrawal-amount",
            "deposit_amount",
            "available_balance",
            "transactionDetails",
        ]

        mapping = detect_column_mapping(columns)
        self.assertEqual(mapping.mapping["date"], "txn_date")
        self.assertEqual(mapping.mapping["transaction_id"], "TransactionID")
        self.assertEqual(mapping.mapping["description"], "transactionDetails")
        self.assertEqual(normalize_column_name(" transaction-ID "), "transaction id")

    def test_common_banking_aliases(self):
        frame = pd.DataFrame({
            "Txn Date": ["3 Jan 2024"],
            "Narration": ["ATM withdrawal"],
            "Withdrawal Amount": [500],
            "Deposit Amount": [""],
            "Closing Balance": [4500],
            "Ref No": ["ABC123"],
        })

        result = standardize_transactions(frame)
        self.assertTrue(result.is_usable)
        self.assertEqual(result.transactions.loc[0, "debit"], 500)
        self.assertEqual(result.transactions.loc[0, "balance"], 4500)

    def test_single_amount_and_transaction_type_layout(self):
        frame = pd.DataFrame({
            "date": ["2024-02-01", "2024-02-02", "2024-02-03", "2024-02-04"],
            "description": ["A", "B", "C", "D"],
            "amount": [100, -200, "300", "400"],
            "transaction_type": ["debit", "CR", "withdrawal", "deposit"],
        })

        result = standardize_transactions(frame)
        self.assertTrue(result.is_usable)
        self.assertEqual(result.transactions["amount"].tolist(), [100.0, 200.0, 300.0, 400.0])
        self.assertEqual(
            result.transactions["transaction_type"].tolist(),
            ["debit", "credit", "debit", "credit"],
        )

    def test_blank_rows_are_reported(self):
        frame = pd.DataFrame({
            "date": ["2024-01-01", ""],
            "description": ["Valid", ""],
            "debit": [100, ""],
            "credit": ["", ""],
        })

        result = standardize_transactions(frame)
        self.assertEqual(result.diagnostics.original_row_count, 2)
        self.assertEqual(result.diagnostics.cleaned_row_count, 1)
        self.assertEqual(result.diagnostics.dropped_row_count, 1)
        self.assertEqual(result.diagnostics.dropped_row_reasons, {"blank_row": 1})

    def test_duplicate_header_row_inside_data_is_reported(self):
        frame = pd.DataFrame({
            "Date": ["01/01/2024", "Date", "02/01/2024"],
            "remarks": ["First", "remarks", "Second"],
            "withdraw": [100, "withdraw", ""],
            "deposits": ["", "deposits", 200],
        })

        result = standardize_transactions(frame)
        self.assertEqual(result.diagnostics.cleaned_row_count, 2)
        self.assertEqual(
            result.diagnostics.dropped_row_reasons,
            {"duplicate_header_row": 1},
        )

    def test_currency_and_comma_numeric_values(self):
        frame = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "description": ["Purchase", "Income"],
            "debit": ["₹1,250.50", ""],
            "credit": ["", "INR 10,000"],
        })

        result = standardize_transactions(frame)
        self.assertEqual(result.transactions["amount"].tolist(), [1250.50, 10000.0])

    def test_optional_transaction_id_and_balance_can_be_absent(self):
        frame = pd.DataFrame({
            "date": ["2024-01-01"],
            "description": ["Purchase"],
            "debit": [50],
        })

        result = standardize_transactions(frame)
        self.assertTrue(result.is_usable)
        self.assertTrue(pd.isna(result.transactions.loc[0, "transaction_id"]))
        self.assertTrue(pd.isna(result.transactions.loc[0, "balance"]))

    def test_missing_required_fields_return_validation_result(self):
        missing_description = pd.DataFrame({
            "date": ["2024-01-01"],
            "debit": [50],
        })
        missing_amount = pd.DataFrame({
            "date": ["2024-01-01"],
            "description": ["Purchase"],
        })

        first = standardize_transactions(missing_description)
        second = standardize_transactions(missing_amount)

        self.assertFalse(first.is_usable)
        self.assertIn("description", first.diagnostics.missing_required_fields)
        self.assertFalse(second.is_usable)
        self.assertIn("amount_and_direction", second.diagnostics.missing_required_fields)
        self.assertEqual(first.diagnostics.dropped_row_reasons, {"schema_validation_failed": 1})

    def test_ambiguous_candidates_are_not_arbitrarily_selected(self):
        frame = pd.DataFrame({
            "Date": ["2024-01-01"],
            "Transaction Date": ["2024-01-01"],
            "Description": ["Purchase"],
            "Debit": [50],
        })

        result = standardize_transactions(frame)

        self.assertFalse(result.is_usable)
        self.assertEqual(
            result.diagnostics.ambiguous_mappings["date"],
            ("Date", "Transaction Date"),
        )
        self.assertIn("date", result.diagnostics.missing_required_fields)
        self.assertEqual(
            result.diagnostics.candidate_columns["date"],
            ("Date", "Transaction Date"),
        )

    def test_manual_mapping_resolves_ambiguity(self):
        frame = pd.DataFrame({
            "Date": ["2024-01-01"],
            "Value Date": ["2024-01-02"],
            "Description": ["Purchase"],
            "Debit": [50],
        })

        result = standardize_transactions(
            frame,
            mapping={
                "date": "Value Date",
                "description": "Description",
                "debit": "Debit",
            },
        )

        self.assertTrue(result.is_usable)
        self.assertEqual(result.diagnostics.date_range, ("2024-01-02", "2024-01-02"))

    def test_read_csv_and_preserve_blank_row_diagnostics(self):
        csv_data = (
            "Date,transactionId,withdraw,deposits,balance,remarks\n"
            "01/01/2024,T1,100,,900,Purchase\n"
            ",,,,,\n"
            "02/01/2024,T2,,200,1100,Refund\n"
        ).encode("utf-8")

        result = read_statement(csv_data, filename="statement.csv")
        self.assertTrue(result.is_usable)
        self.assertEqual(result.diagnostics.original_row_count, 3)
        self.assertEqual(result.diagnostics.dropped_row_reasons, {"blank_row": 1})

    def test_read_xlsx(self):
        frame = pd.DataFrame({
            "Txn Date": ["2024-01-01"],
            "Narration": ["Salary"],
            "Withdrawal Amount": [""],
            "Deposit Amount": [10000],
        })
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False)

        result = read_statement(buffer.getvalue(), filename="statement.xlsx")
        self.assertTrue(result.is_usable)
        self.assertEqual(result.transactions.loc[0, "credit"], 10000)


if __name__ == "__main__":
    unittest.main()
