# FinPulse 2.0

FinPulse 2.0 is a behavioral-finance analytics system that turns CSV or XLSX bank statements into an explainable, personalized financial-behavior report. It combines flexible statement ingestion, reviewable rule-based categorization, transparent scoring, behavioral signals, and practical recommendations while keeping uploaded data in the active Streamlit session.

Traditional payment and UPI applications primarily show transaction history and basic spending summaries. FinPulse goes further by interpreting how a person's reviewed spending relates to the monthly funds they say are available, while keeping the calculations inspectable and the limitations explicit.

> FinPulse provides behavioral analytics for informational and educational purposes and is not financial advice.

## What FinPulse 2.0 Does

```text
Upload CSV/XLSX statement
          ↓
Normalize statement
          ↓
Auto-categorize transactions
          ↓
User review, corrections, and exclusions
          ↓
Behavioral feature engineering
          ↓
Explainable scoring
          ↓
Signals and recommendations
          ↓
Personalized report
```

The original FinPulse v1 synthetic-user dashboard and its frozen segmentation assets remain available as a separate demonstration of the earlier analytical workflow.

## Key Features

- Flexible CSV and XLSX statement ingestion
- Case-insensitive, alias-aware column detection with manual mapping fallback
- Layered merchant, phrase, keyword, context, and conservative fuzzy categorization
- High, Medium, and Low categorization confidence
- Transaction review and category correction before analysis
- Explicit inclusion or exclusion of exceptional transactions
- User-declared Monthly Available Amount as the behavioral denominator
- Optional Monthly Spending Budget and budget-utilization reporting
- Explainable rule-based FinPulse Behavioral Score
- Behavioral signals and deterministic recommendations
- Session-only upload processing with no upload-workflow writes to SQLite
- Privacy-safe synthetic statements for testing and demonstration

## Supported Input

FinPulse accepts CSV and XLSX bank statements. PDF parsing is future work.

Bank formats vary, so column names are normalized case-insensitively and matched against common aliases. The priority real-world-style schema is:

```text
Date
transactionId
withdraw
deposits
balance
remarks
```

Variants such as `Transaction ID`, `Txn ID`, `Debit Amount`, `Credit Amount`, `Narration`, and single `Amount` plus debit/credit indicator layouts are also supported. Transaction ID and balance are optional. If automatic detection is ambiguous, the Analyze Statement page provides manual column mapping.

## Monthly Available Amount

Monthly Available Amount is required and is not salary-only. It may represent:

- salary
- stipend
- pocket money
- household allowance
- other monthly funds available

FinPulse uses this user-declared amount as financial capacity instead of treating every statement credit as income. Bank credits can include reimbursements, transfers, refunds, loans, or money temporarily passing through an account, so using all credits as verified recurring income would make behavioral ratios misleading.

## Statement Cash Flow vs. Behavioral Analysis

The report deliberately separates two views:

- **Statement cash flow** reports observed credits, observed debits, and net statement cash flow for the reviewed full statement. It describes what happened in the account.
- **Spending considered for analysis** is the total of included debit transactions used by FinPulse behavioral analytics.

Observed credits remain visible as statement information, but credits do not drive uploaded-user spending ratios, savings/remaining calculations, behavioral score, signals, recommendations, or K-Means feature generation. Excluded debits remain visible in the statement review and cash-flow summary but do not affect behavioral analytics.

## Transaction Categorization

Categorization is deterministic and layered:

1. Transaction direction and context
2. Known merchant rules
3. Strong phrases
4. Generic keywords
5. Conservative fuzzy matching
6. `Others` fallback

Detailed labels such as Grocery or Food Delivery are first inferred, then mapped to the six FinPulse categories:

- Income
- Essentials
- Desire
- Repayment
- Investment/Savings
- Others

Merchant descriptions can be incomplete or ambiguous, so classification is not assumed to be perfect. Users can review every transaction, focus on Low-confidence rows, correct `final_category`, and exclude exceptional transactions before confirming the report input. The original predicted category remains preserved for auditability.

## Explainable Behavioral Score

FinPulse uses separate rule-based score contracts for its two data paths. The
scores are educational heuristics and have not been clinically, financially,
or outcome validated.

### FinPulse v1 synthetic-user score

| Component | Maximum points |
| --- | ---: |
| Spending Control | 30 |
| Savings | 25 |
| Debt Management | 25 |
| Stability | 20 |
| **Total** | **100** |

This original four-component contract remains unchanged for the synthetic-user
dashboard.

### FinPulse 2.0 uploaded-statement score

| Component | Subcomponents | Maximum points |
| --- | --- | ---: |
| Spending Control | Overall spending 20 + Desire control 12 + spending discipline 8 | 40 |
| Saving & Investment Behavior | Explicit Investment/Savings 20 + estimated unspent capacity 15 | 35 |
| Repayment Management | Monthly repayment burden | 25 |
| **Total** |  | **100** |

The upload score uses Monthly Available Amount and included debit transactions
with reviewed `final_category` values. Arbitrary statement credits are not
reliable evidence of recurring income, so credits do not enter this score. The
upload score therefore has no Stability component and does not renormalize an
incomplete v1 score.

The transparent thresholds are:

- Overall spending ratio points: `≤60%: 20`, `≤75%: 18`, `≤90%: 14`,
  `≤100%: 9`, `≤110%: 4`, `>110%: 0`.
- Desire ratio points: `≤5%: 12`, `≤10%: 10`, `≤15%: 7`, `≤25%: 3`,
  `>25%: 0`.
- Overspending-month discipline points: `0%: 8`, `≤15%: 6`, `≤30%: 4`,
  `≤50%: 2`, `>50%: 0`.
- Explicit Investment/Savings ratio points: `≥20%: 20`, `≥15%: 17`,
  `≥10%: 14`, `≥5%: 9`, `>0%: 4`, `0%: 0`.
- Estimated unspent-capacity ratio points: `≥30%: 15`, `≥20%: 12`,
  `≥10%: 9`, `≥0%: 5`, `<0%: 0`.
- Repayment ratio points: `≤5%: 25`, `≤10%: 23`, `≤20%: 19`,
  `≤30%: 13`, `≤40%: 7`, `>40%: 3`.

Investment/Savings transactions are included in total debit spending. Their
explicit-allocation points recognize observable saving behavior, while the
estimated remainder is calculated after all included debits, so the same
transaction does not also inflate the remainder. Estimated remainder is not
presented as confirmed savings.

When a monthly budget is supplied, budget utilization refines only the
eight-point spending-discipline subcomponent: `≤90%: 8`, `≤100%: 6`,
`≤110%: 3`, `>110%: 0`. The stricter of budget adherence and observed
capacity-overspending points is used. Without a budget, all eight points remain
available from observed overspending behavior, so omitting a budget creates no
structural score penalty.

Both score paths retain the existing bands: **Strong** `80–100`, **Stable**
`65–79`, **Watchful** `50–64`, **Strained** `35–49`, and **High Pressure**
`0–34`. Report Confidence remains separate and can be Low even when the upload
score is calculated out of 100.

## K-Means / ML

FinPulse v1 includes a frozen K-Means behavioral segmentation prototype fitted on:

- 1,000 synthetic reference users
- eight behavioral features
- five clusters (`k=5`)

The fitted scaler, K-Means model, feature order, and persona mapping are stored in `models/finpulse_kmeans_v1.joblib`. The artifact is preserved for reproducibility, and the synthetic dashboard demonstrates the five interpreted personas. This is not production-validated customer segmentation.

Uploaded-user persona inference is intentionally withheld. The frozen model requires `income_cv`, but that feature cannot be derived defensibly when arbitrary bank credits are not assumed to be recurring income. Showing no persona is a model-validity safeguard, not a runtime failure.

## Architecture

```text
Uploaded statement
        ↓
    Ingestion
        ↓
 Standardization
        ↓
 Categorization
        ↓
   User review
        ↓
Included debit transactions ───────────────┐
        ↓                                  │
Feature engineering                       │
        ↓                                  │
Rule-based score / signals / recommendations
        ↓
      Report

Synthetic reference users
        ↓
SQL + Python features
        ↓
Frozen scaler + K-Means (k=5)
        ↓
Synthetic dashboard personas
```

## Repository Structure

```text
app/        Streamlit entrypoint, original dashboard, and Analyze Statement page
finpulse/   Ingestion, categorization, review, upload analytics, reporting, inference
scripts/    Original synthetic-data, feature, score, signal, and model tooling
sql/        SQL feature-engineering pipeline for synthetic/reference users
models/     Frozen v1 segmentation artifact
tests/      Unit, integration, reconciliation, and synthetic statement fixtures
data/       Original reference and validation assets
database/   Bundled FinPulse v1 SQLite database
notebooks/  Exploratory analysis and clustering rationale
requirements-dev.txt  Test-only dependencies layered on production requirements
```

The upload path is in-memory and separate from the SQLite-backed synthetic dashboard. Do not run `scripts/run_pipeline.py` merely to launch the application; the required v1 database and frozen model are already bundled.

## Demo

The preferred privacy-safe demo uses [tests/fixtures/statements/student_1month.xlsx](tests/fixtures/statements/student_1month.xlsx). All bundled statement fixtures are deterministic synthetic data; they do not contain real bank or customer information.

Use:

- Monthly Available Amount: **₹5,000**
- Monthly Spending Budget: **₹4,500**

During review:

1. Change `STU-AMAZON` from `Others` to `Essentials` for the demo assumption that it represents study or household supplies.
2. Exclude `STU-TEMP-CREDIT`.
3. Exclude `STU-TEMP-DEBIT`.
4. Confirm the reviewed transactions.
5. Generate the report.

Expected synthetic benchmark values are approximately:

| Metric | Expected value |
| --- | ---: |
| Spending considered for analysis | ₹7,273 |
| Essentials | ₹3,975 |
| Desire | ₹2,488 |
| Others | ₹810 |
| Expense ratio | 145.46% |
| Budget utilization | 161.62% |

Observed statement cash flow still includes the synthetic temporary credit/debit pair because cash flow describes the full statement, while behavioral analytics exclude the reviewed exceptional transactions.

## Running Locally

Python 3.13 is the currently validated environment. From the repository root:

```bash
python3 -m venv fenv
source fenv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app/app.py
```

Open the **Analyze Statement** page from Streamlit navigation to use the FinPulse 2.0 upload workflow. The main page preserves the original synthetic-user dashboard. The analytical v1 variant can also be launched directly with `streamlit run app/app_v1.py`.

`fenv/` is local-only and must not be committed.

## Testing

Run the complete suite with the project-local interpreter:

```bash
./fenv/bin/python -m pip install -r requirements-dev.txt
./fenv/bin/python -m pytest
```

Current release validation: **170 passed**.

## Privacy and Security

- Uploaded statement bytes and reviewed transactions are held in Streamlit session state for the active session.
- The upload workflow does not write statement data to `database/finpulse.db` or other repository files.
- The bundled database is used by the preserved synthetic-user dashboard.
- Included statement fixtures are synthetic and privacy-safe.
- FinPulse does not claim end-to-end encryption or bank-grade data custody controls.

For deployment, use platform secret management if future integrations require credentials. Do not commit `.env` or `.streamlit/secrets.toml` files.

## Limitations

- CSV and XLSX only; PDF statements are not yet supported.
- Bank formats vary, and some may require manual column mapping.
- Merchant descriptions can be incomplete or ambiguous; users should review categories.
- The rule-based score is heuristic and not calibrated against real financial outcomes.
- Uploaded-user Report Confidence may be Low for short or low-volume statements,
  independently of the complete 100-point behavioral score.
- The K-Means reference population is synthetic and not production-validated.
- Uploaded-user K-Means personas are intentionally withheld due to feature incompatibility.
- No production bank API or Open Banking integration is implemented.
- Uploaded data is session-only, so a browser/session reset loses the current review.
- FinPulse is an educational portfolio project, not financial advice.

## Future Work

- PDF statement parsing
- Wider bank-format compatibility
- Richer merchant metadata
- Anonymized real-world model validation
- Optional bank API or Open Banking integration
- Improved categorization with richer transaction metadata and carefully evaluated NLP

## Technology Stack

Python, Pandas, NumPy, SQLite, SQL, scikit-learn, Plotly, Matplotlib, Streamlit, OpenPyXL, RapidFuzz, and Joblib.
