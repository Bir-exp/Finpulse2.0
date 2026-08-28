from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from finpulse.categorization import categorize_transactions
from finpulse.presentation import format_inr
from finpulse.reporting import REPORT_KEY, generate_financial_report
from finpulse.review import initialize_review_categories
from finpulse.statement_ingestion import read_statement


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = PROJECT_ROOT / "app" / "pages" / "1_Analyze_Statement.py"
FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "statements" / "salaried_3months.xlsx"
)


class AnalyzeStatementPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ingestion = read_statement(
            FIXTURE_PATH.read_bytes(), filename=FIXTURE_PATH.name
        )
        cls.reviewed = initialize_review_categories(
            categorize_transactions(ingestion.transactions)
        )
        cls.report = generate_financial_report(
            cls.reviewed,
            72_000,
            60_000,
            review_confirmed=True,
        )

    def test_generated_report_uses_clean_tabs_and_monthly_values(self):
        app = AppTest.from_file(str(PAGE_PATH))
        app.session_state[REPORT_KEY] = self.report
        app.session_state["categorized_transactions"] = self.reviewed
        app.session_state["reviewed_transactions"] = self.reviewed.copy(deep=True)
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            [
                "Overview",
                "My Spending",
                "Improve",
                "Transactions",
                "Advanced Analysis",
            ],
        )

        overview_metrics = {metric.label: metric.value for metric in app.tabs[0].metric}
        statement = self.report.analytics.statement_summary
        self.assertEqual(
            overview_metrics["Average Monthly Spending"],
            format_inr(statement["monthly_normalized_debit_spending"]),
        )
        self.assertNotEqual(
            overview_metrics["Average Monthly Spending"],
            format_inr(statement["total_debit_spending"]),
        )
        self.assertIn(
            "PROVISIONAL", " ".join(caption.value for caption in app.tabs[0].caption)
        )

        overview_text = " ".join(
            [
                *(metric.label for metric in app.tabs[0].metric),
                *(caption.value for caption in app.tabs[0].caption),
                *(markdown.value for markdown in app.tabs[0].markdown),
            ]
        )
        for technical_name in (
            "avg_expense_ratio",
            "income_cv",
            "expense_ratio_volatility",
            "cluster distance",
        ):
            self.assertNotIn(technical_name, overview_text)

        spending_metrics = {metric.label: metric.value for metric in app.tabs[1].metric}
        cash_flow = self.report.analytics.statement_cash_flow_summary
        self.assertEqual(
            spending_metrics["Observed Credits"],
            format_inr(cash_flow["observed_credits"]),
        )
        self.assertEqual(
            spending_metrics["Observed Debits"],
            format_inr(cash_flow["observed_debits"]),
        )

        advanced_labels = {metric.label for metric in app.tabs[4].metric}
        self.assertIn("Expense ratio", advanced_labels)
        self.assertIn("Spending Control", advanced_labels)
        self.assertIn("Stability", advanced_labels)
        self.assertIn("Report Confidence", advanced_labels)

        transaction_text = " ".join(
            markdown.value for markdown in app.tabs[3].markdown
        )
        self.assertIn("transactions included in analysis", transaction_text)


if __name__ == "__main__":
    unittest.main()
