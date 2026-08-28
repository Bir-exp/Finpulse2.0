# Synthetic statement fixtures

These files are deterministic, privacy-safe test data for FinPulse 2.0. All
names, references, UPI handles, balances, and transactions are fictional. No
real bank statement or user financial data was used.

The machine-readable source of expected mappings, selected category outcomes,
exceptional transactions, row diagnostics, and demo benchmarks is
`manifest.json`.

## Fixture set

| File | Scenario | Source rows | Format |
| --- | --- | ---: | --- |
| `student_1month.xlsx` | Student/pocket money with a temporary parent transfer | 33 | Exact FinPulse priority schema |
| `salaried_3months.xlsx` | Salaried user with recurring costs, debt, and investments | 75 | Exact FinPulse priority schema |
| `household_3months.csv` | Non-salaried household with irregular credits and controlled dirty rows | 35 | Optional ID/balance omitted |
| `mixed_6months.xlsx` | Six-month normalization, trend, and volume stress case | 128 | Common bank aliases |
| `alternate_bank_format.csv` | Alternate bank column-name compatibility | 16 | `Txn Date`, `Ref No`, narration and withdrawal/deposit aliases |
| `amount_type_format.xlsx` | Single amount plus direction indicator | 18 | `Debit`, `Credit`, `DR`, and `CR` |
| `synthetic_metadata_statement.xls` | Legacy Excel fixture with fake metadata before the table | 3 | `.xls`, header row detection, plural withdrawal alias |

## Preferred manual demo

Use `student_1month.xlsx` with:

- Monthly Available Amount: ₹5,000
- Monthly Spending Budget: ₹4,500
- Correct `STU-AMAZON` from `Others` to `Essentials` for the demo assumption
  that it represents study/household supplies.
- Exclude `STU-TEMP-CREDIT` and `STU-TEMP-DEBIT`; they are a temporary ₹20,000
  parent transfer and matching payment made on the parent's behalf.
- Confirm the reviewed transactions and generate the report.

Expected benchmark after those review decisions:

- Spending considered for analysis: ₹7,273
- Essentials: ₹3,975
- Desire: ₹2,488
- Others: ₹810
- Repayment and Investment/Savings: ₹0
- Expense ratio: 1.4546
- Budget utilization: 161.62%

Observed statement cash flow intentionally still includes the temporary
credit/debit pair. Uploaded-user Stability and K-Means persona remain
unavailable by product design.
