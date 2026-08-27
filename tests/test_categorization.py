import unittest

import pandas as pd

from finpulse.categorization import (
    ALLOWED_CATEGORIES,
    categorize_transaction,
    categorize_transactions,
    clean_description,
    extract_receiver,
    load_category_rules,
)


class CategorizationTests(unittest.TestCase):
    def assert_category(self, description, transaction_type, detailed, category, confidence=None):
        result = categorize_transaction(description, transaction_type)
        self.assertEqual(result["detailed_category"], detailed)
        self.assertEqual(result["predicted_category"], category)
        if confidence is not None:
            self.assertEqual(result["confidence"], confidence)
        return result

    def test_swiggy_food_delivery_high_confidence(self):
        self.assert_category("UPI/624834720195/SWIGGY/ICICI/sw*****@ybl", "debit", "Food Delivery", "Desire", "High")

    def test_zomato_is_desire(self):
        self.assert_category("ZOMATO LIMITED", "debit", "Food Delivery", "Desire", "High")

    def test_local_grocery_is_essential(self):
        self.assert_category("MAA TARINI GROCERY", "debit", "Groceries", "Essentials")

    def test_kirana_is_essential(self):
        self.assert_category("SHREE KRISHNA KIRANA", "debit", "Groceries", "Essentials")

    def test_stationery_is_essential(self):
        self.assert_category("ABC STATIONERY", "debit", "Education/Stationery", "Essentials")

    def test_barber_is_personal_care(self):
        self.assert_category("RAJU BARBER", "debit", "Personal Care", "Desire")

    def test_apollo_pharmacy_is_healthcare(self):
        self.assert_category("APOLLO PHARMACY", "debit", "Healthcare", "Essentials", "High")

    def test_credit_card_payment_is_repayment(self):
        self.assert_category("ACH D- HDFC BANK CREDIT CARD PAYMENT", "debit", "Credit Card Payment", "Repayment", "High")

    def test_loan_emi_is_repayment(self):
        self.assert_category("HDFC LOAN EMI PAYMENT", "debit", "Loan EMI", "Repayment", "High")

    def test_zerodha_is_investment(self):
        self.assert_category("ZERODHA BROKING", "debit", "Brokerage/Investment", "Investment/Savings", "High")

    def test_groww_sip_is_investment(self):
        self.assert_category("UPI/GROWW SIP/AXIS/groww@ybl", "debit", "Mutual Fund/SIP", "Investment/Savings", "High")

    def test_salary_credit_is_income(self):
        self.assert_category("NEFT ACME TECHNOLOGIES SALARY", "credit", "Salary", "Income", "High")

    def test_person_to_person_credit_is_not_income(self):
        result = self.assert_category(
            "UPI/674839201271/RAHUL KUMAR/AXIS/rahul@okaxis",
            "credit",
            "Unknown Transfer",
            "Others",
            "Low",
        )
        self.assertEqual(result["receiver"], "Rahul Kumar")

    def test_unknown_person_debit_is_others(self):
        self.assert_category("UPI TO PRIYA SHARMA", "debit", "Unknown Transfer", "Others", "Low")

    def test_ambiguous_amazon_is_not_high_confidence(self):
        result = categorize_transaction("AMAZON PAY PURCHASE", "debit")
        self.assertEqual(result["predicted_category"], "Others")
        self.assertIn(result["confidence"], {"Medium", "Low"})
        self.assertNotEqual(result["confidence"], "High")

    def test_swigy_spelling_variant_fuzzy_matches(self):
        self.assert_category("SWIGY", "debit", "Food Delivery", "Desire", "Medium")

    def test_strong_context_overrides_merchant_keyword(self):
        self.assert_category(
            "HDFC CREDIT CARD PAYMENT DMART",
            "debit",
            "Credit Card Payment",
            "Repayment",
            "High",
        )

    def test_null_and_empty_descriptions_do_not_crash(self):
        for description in (None, "", pd.NA):
            with self.subTest(description=description):
                result = categorize_transaction(description, "debit")
                self.assertEqual(result["predicted_category"], "Others")
                self.assertEqual(result["confidence"], "Low")

    def test_receiver_extraction_examples(self):
        examples = {
            "UPI-SWIGGY-SWIGGY@YBL": "Swiggy",
            "UPI/674839201271/RAHUL KUMAR/AXIS/rahul@okaxis": "Rahul Kumar",
            "ACH D- HDFC BANK CREDIT CARD": "Hdfc Bank Credit Card",
            "NEFT ACME TECHNOLOGIES SALARY": "Acme Technologies Salary",
        }
        for description, expected in examples.items():
            with self.subTest(description=description):
                self.assertEqual(extract_receiver(description), expected)

    def test_all_outputs_use_allowed_categories_and_preserve_row_count(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "transaction_id": ["T1", "T2", "T3"],
            "debit": [200.0, 0.0, 50.0],
            "credit": [0.0, 5000.0, 0.0],
            "balance": [800.0, 5800.0, 5750.0],
            "description": ["SWIGGY", "SALARY CREDIT", "UNKNOWN SHOP"],
            "amount": [200.0, 5000.0, 50.0],
            "transaction_type": ["debit", "credit", "debit"],
        })

        result = categorize_transactions(frame)

        self.assertEqual(len(result), len(frame))
        self.assertTrue(set(result["predicted_category"]).issubset(ALLOWED_CATEGORIES))
        self.assertTrue({"receiver", "detailed_category", "predicted_category", "confidence"}.issubset(result.columns))
        self.assertNotIn("final_category", result.columns)

    def test_credit_refund_is_distinguished_from_salary(self):
        self.assert_category("ZOMATO ORDER REFUND", "credit", "Refund", "Income", "High")

    def test_debit_salary_phrase_is_never_income(self):
        result = categorize_transaction("SALARY ADVANCE PAYMENT", "debit")
        self.assertNotEqual(result["predicted_category"], "Income")

    def test_rule_configuration_is_valid(self):
        rules = load_category_rules()
        self.assertEqual(set(rules["allowed_categories"]), set(ALLOWED_CATEGORIES))
        self.assertGreaterEqual(rules["fuzzy_threshold"], 90)

    def test_clean_description_preserves_content(self):
        self.assertEqual(clean_description("  SWIGGY\nLIMITED  "), "SWIGGY LIMITED")


if __name__ == "__main__":
    unittest.main()
