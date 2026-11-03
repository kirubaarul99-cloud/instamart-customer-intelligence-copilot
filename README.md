# Overview

This repository implements an offline **Instamart Customer Intelligence & Support Copilot** using fictional, reproducible data. It connects four stages: (A) clean exports and store them in CSV/SQLite, (B) create RFM customer segments and a leakage-aware lapse model, (C) redact and triage support tickets, and (D) answer policy questions and run a bounded refund-support agent.

All records are synthetic. `generate_data.py` and `mock_llm.py` are copied verbatim from the supplied capstone brief and must remain unchanged. The mock model is a deterministic, rule-based teaching stand-in; no API key, hosted model, paid service, or network request is used.

# Reproduce

Python 3.11 or newer is recommended. From the repository root, run the following commands in order. They use a local virtual environment and regenerate the required artifacts.

**Windows PowerShell**

1. `python -m venv .venv`
2. `.venv\Scripts\python.exe -m pip install --upgrade pip`
3. `.venv\Scripts\python.exe -m pip install -r requirements.txt`
4. `.venv\Scripts\python.exe generate_data.py`
5. `.venv\Scripts\python.exe run_project.py`
6. `.venv\Scripts\python.exe run_sql_analysis.py`
7. `.venv\Scripts\python.exe run_triage_experiments.py`
8. `.venv\Scripts\python.exe run_policy_evaluation.py`
9. `.venv\Scripts\python.exe run_agent_scenarios.py`

On macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python` in commands 2–9. The generator's final line should be `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The validation stage prints `ALL CHECKPOINTS PASSED` when the Part A checks hold. `run_project.py` creates the clean CSVs, SQLite database, customer model artifacts, full few-shot triage output, and standard-overlap policy evaluation. The following scripts run the SQL/EDA report, zero-shot and budget experiments, zero-overlap policy evaluation, and all refund-agent demonstrations.

## Notebook Guide

Two project-workflow notebooks are included in `notebooks/`:

- `Part_AB_Customer_Intelligence.ipynb` — Part A and Part B: raw-data audit, cleaning, SQLite/SQL analysis, EDA, RFM segmentation, and lapse-model evaluation.
- `Part_CD_Support_Copilot.ipynb` — Part C and Part D: ticket triage, privacy and budget checks, policy retrieval, refund-agent scenarios, and safety demonstrations.

Run the Part A & B notebook first, then Part C & D. The notebooks use this repository's `src/` modules for the implementation and provide a student-style workflow for running and inspecting the experiments. Run them from inside the project directory after installing `requirements.txt` in Python 3.11 or newer.

# Part A

## Raw-data audit

| Export | Shape | Column types (raw pandas read) | Blank/missing values in raw export |
|---|---:|---|---|
| `customers.csv` | 1,200 × 5 | All five columns read as `object` | `age`: 61 (5.08%); `gender`: 30 (2.50%); other columns: 0 |
| `products.csv` | 40 × 4 | IDs/name/category: `object`; `price_inr`: `int64` | All columns: 0 |
| `orders.csv` | 13,084 × 9 | IDs, timestamp, status, amount and delivery: `object`; quantity and discount: `int64` | `delivery_minutes`: 2,541 (19.42%); other columns: 0 blank strings. `order_amount` also has 120 non-numeric values (`N/A`/`error`). |

The raw-order checks found 119 exact duplicate rows, 656 `pending` rows, 12,309 distinct order IDs, 120 non-numeric amounts, 106 negative delivery times, and 2,541 blank delivery times. Customer checks found 61 blank ages, 6 ages outside 10–100, 24 raw city spellings, 10 distinct non-blank raw gender spellings, and 30 blank genders.

## Cleaning rules and results

- **Customers:** trim city names and use title case; map gender variants to `male`, `female`, or `unknown`; convert ages outside 10–100 to missing and fill all missing ages with the median age in the customer's city using `groupby(...).transform`; parse signup dates. The final city counts are Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Gender counts are female 610, male 560, unknown 30. A total of 67 ages (61 blank and 6 out of range) were imputed; there are no missing ages and all cleaned ages are 18–60.
- **Orders:** parse timestamps; remove 119 exact duplicate rows; for remaining duplicate order IDs keep the latest timestamp, removing 656 additional rows; convert amounts to numeric; reconstruct 112 missing/unparseable amounts using `price × quantity × (1 − discount_pct/100)`; turn 106 negative delivery values into missing; fill 774 missing delivery values for completed orders with the median completed delivery time for the product category; keep delivery time missing for cancelled orders. The final data has 12,309 unique orders: 11,110 completed and 1,199 cancelled. Completed orders have no missing amount or delivery time; cancelled orders have 1,199 missing delivery times. Completed-order revenue totals **INR 2,998,668.80**.
- **Storage and validation:** `data/clean/customers.csv`, `products.csv`, and `orders.csv` are written to disk and loaded into `data/instamart.db` tables named `customers`, `products`, and `orders`. The validation function checks the clean table sizes, key uniqueness, allowed values, delivery rules, and revenue total.

## SQL analysis

The seven queries in `sql/queries.sql` are executed against the SQLite database by `run_sql_analysis.py`.

| Query | Result |
|---|---|
| Q1 — `WHERE` then `HAVING` | 30 customers |
| Q2 — city × category revenue join | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics at INR 233,909.45 |
| Q3 — products above average catalogue price | 11 products |
| Q4 — customers with no completed order | 25 customers |
| Q5 — CTE for spend above average | 550 customers; average completed spend INR 2,552.06 |
| Q6 — `RANK`, `DENSE_RANK`, `ROW_NUMBER` | 1,175 customers with completed spend; 8 rows have `RANK()` different from `ROW_NUMBER()`. Ties share a rank, while `ROW_NUMBER()` still assigns a unique position. Top spenders: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10). |
| Q7 — daily running total | 183 days; final cumulative revenue INR 2,998,668.80 |

## Pandas cross-check and EDA

The `pd.pivot_table` cross-check of revenue by city and product category has a grand total of INR 2,998,668.80, matching SQL.

| City | Completed revenue (INR) | Mean order amount (INR) | Completed orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

For completed orders, mean order amount is INR 269.91 and median is INR 135.00; skewness is 4.61. Q1 is INR 64.80, Q3 is INR 297.00, and the 1.5×IQR fences are −INR 283.50 and INR 645.30. There are 0 observations below the lower fence and 897 above the upper fence.

**Business interpretation (EDA):** These data describe fictional grocery and retail orders across six cities, joined to customer and product attributes. Order amounts are strongly right-skewed: the INR 269.91 mean is much higher than the INR 135.00 median, with 897 high-value outliers above INR 645.30. These values may reflect expensive products or larger baskets, so they should be inspected rather than automatically removed. The strongest absolute correlation among amount, discount, quantity and delivery time is order amount versus discount percentage (about −0.095); this small correlation does not establish causation.

# Part B

## RFM and customer segments

RFM is calculated only from completed orders at snapshot `2025-10-01 00:00:00`. It includes 1,175 customers; recency median is 19 days (minimum 0, maximum 180), mean frequency is 9.46 (maximum 33), mean monetary value is INR 2,552.06 (maximum INR 14,766.55).

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.76 | 0.465 |
| 3 | 1,268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

We choose **K=4**: the inertia drop from K=3 to K=4 is substantial and K=4 has the strongest silhouette among the tested choices, while four segments are still understandable for campaign planning. Inertia always falls as K increases because more centroids can fit the observations more closely; that alone cannot identify a useful number of groups because it rewards adding clusters even when the extra groups add little business value.

| Segment ID | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Name | Recommended action |
|---:|---:|---:|---:|---:|---|---|
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Occasional | Send low-cost replenishment reminders and relevant offers. |
| 1 | 138 | 24.12 | 6.77 | 7,129.70 | Premium | Offer tailored service and high-value retention benefits. |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 | Frequent | Encourage repeat baskets with convenience-focused recommendations. |
| 3 | 182 | 109.04 | 2.20 | 685.99 | Dormant | Use a reactivation message and monitor response before discounting further. |

Cluster IDs are arbitrary labels, not an ordinal ranking. Premium is assigned to the segment with the highest mean monetary value. PCA on scaled RFM has explained-variance ratios **0.634** and **0.218** (total about 0.852). The segment output is `outputs/customer_segments.csv`.

## Lapse prediction

The cohort includes 1,156 customers with completed orders before 2025-09-01; 443 (38.32%) lapsed because they had no completed order during September. A stratified split produced 924 training and 232 test rows. The feature matrix uses only pre-cutoff order history plus customer attributes; the September window is used only for the label. Numeric scaling and one-hot encoding are inside pipelines fitted on training rows only; no encoder, scaler or resampler saw test rows, and no segment label is a feature. The post-imputation feature matrix has zero missing values. The fitted training scaler's mean frequency is 8.115 versus 8.029 for the full cohort because the training split is only a subset of the cohort.

The all-negative baseline has 61.64% accuracy but lapsed-class precision, recall and F1 are all 0.00, so it fails to identify any lapsed customers.

| Model | Best parameters | CV F1 | Test confusion matrix `[[TN, FP], [FN, TP]]` | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic Regression | `C=100` | 0.651 | `[[111, 32], [31, 58]]` | 0.644 | 0.652 | 0.648 |
| Random Forest | `max_depth=8`, `min_samples_leaf=10` | 0.687 | `[[110, 33], [32, 57]]` | 0.633 | 0.640 | 0.637 |

Both models exceed the required test F1 of 0.55. We select Random Forest for its stronger cross-validated F1 and its ability to represent non-linear relationships; its test F1 is 0.637. Missing a lapsing customer can mean losing future revenue, while wrongly targeting a loyal customer wastes campaign spend, so recall and precision should both be monitored.

Top five Random Forest feature importances are monetary (0.280), frequency (0.278), recency (0.109), average delivery time (0.060), and cancellation rate (0.052). In Logistic Regression, recency has a positive coefficient (+0.039), frequency a negative coefficient (−1.697), and monetary a positive coefficient (+0.283): in this fitted model, older last-order recency and higher monetary value are associated with a higher predicted lapse log-odds, while greater frequency is associated with lower lapse log-odds, conditional on other features.

The training-only oversampling experiment duplicated lapsed training rows with replacement until both classes had 570 rows. On the unchanged test set, balanced Logistic Regression achieved precision 0.589, recall 0.854, and F1 0.697, compared with the unbalanced Logistic Regression's precision 0.644, recall 0.652, and F1 0.648. Recall moved most: more lapsed customers were found, at the cost of more false-positive retention targets. The saved model is reloaded from `outputs/lapse_model.joblib`; its predictions exactly match the original model's test predictions. The top 10 Premium test-set customers by predicted lapse probability are written to `outputs/retention_list.csv`; 1 of the 10 truly lapsed.

# Part C

The ticket pipeline processes all 164 synthetic tickets in file order. Empty text is skipped and inputs over 400 estimated tokens are flagged before any model call. Email, phone and order-ID patterns are redacted in that order and asserted absent from the prompt. The full run redacted 55 emails, 91 phone numbers and 132 order IDs (278 substitutions total). Transient failures are retried up to three times with exponential backoff and jitter; the 19 simulated transient errors were recovered.

| Prompt | Classified tickets | Accuracy on classified tickets | Total response tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% (126/147) | 7,372 | 50.15 |
| Few-shot | 147 | 100% (147/147) | 12,520 | 85.17 |

The zero-shot confusion table (true category by predicted category) is:

| True \\ Predicted | Billing | Delivery | Fraud | Technical |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Fraud | 11 | 0 | 23 | 0 |
| Technical | 10 | 0 | 0 | 29 |

All zero-shot errors are fraud tickets predicted as billing (11) or technical tickets predicted as billing (10). Few-shot adds about 35.02 tokens per classified ticket but improves accuracy by 14.3 percentage points in this deterministic experiment, so the additional prompt cost is worthwhile when avoiding misrouted fraud and technical tickets is important.

The final few-shot run produced 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `skipped_empty`, and 4 `flagged_long` rows. The zero-shot run with a token budget of 6,000 saved `outputs/triage_budget_demo.csv`; 29 tickets were marked `budget_exhausted`. `outputs/triage_results.csv` contains all 164 rows. JSON replies with an allowed category and keyword-extracted replies with exactly one allowed category were rule-validated; empty, long, malformed/invalid, and budget-exhausted tickets went to human review. Fraud tickets went to the security team with HIGH priority, Premium customers to the premium desk with HIGH priority, and other valid tickets to the support queue.

# Part D

## Policy RAG

Five policy files are split into 40-word chunks with 10-word overlap, giving 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2. `TfidfVectorizer()` represents each chunk; term frequency reflects how often a term appears in a passage, while inverse document frequency gives less weight to terms common across many passages. TF-IDF is simple and inspectable but can miss semantically equivalent phrases that use different words.

| Chunk configuration | Expected document retrieved | Expected fact found in answer | Correct fallback on unanswerable questions |
|---|---:|---:|---:|
| 40 words, overlap 10 | 12/12 | 12/12 | 2/2 |
| 40 words, overlap 0 | 12/12 | 10/12 | 2/2 |

With no overlap, the questions “What delivery time makes a completed order eligible for a full refund?” and “How soon is money returned when a payment fails but is deducted?” fail the expected-fact substring check because the relevant policy sentence is split across chunk boundaries. The assistant answers only from retrieved context and falls back with “I could not find…” when the answer is not in the supplied policy chunks.

## Refund tools and ReAct agent

Five tools are available in `TOOLS_SPEC` and `TOOL_MAP`: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. The tool functions read the SQLite database, retrieve policy chunks, calculate full refunds only for completed orders delivered after 45 minutes, issue refunds of INR 1,000 or less, and record higher-value cases for human approval. The JSONL ledger is a local simulation, not a payment integration.

| Scenario | Status | Refund amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 882 |
| S2 | REFUND | 135.00 | 5 | 882 |
| S3 | ESCALATE | 1,999.00 | 5 | 869 |
| S4 | DENIED | 0.00 | 4 | 593 |
| S5 | DENIED | 0.00 | 3 | 328 |
| S6 | NOT_FOUND | 0.00 | 2 | 66 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 2,982 |

S1 with `token_limit=200` ends with **FALLBACK_TOKEN_LIMIT** (3 model calls; the tool requested by the call that crosses the limit is not executed). S1 with `time_limit_seconds=0` ends with **FALLBACK_TIMEOUT** before any model call. In the crash-and-resume demonstration, a simulated crash occurs after the second tool has been checkpointed. The ordered tool executions across the interrupted and resumed run are exactly `get_order → search_policy → calculate_refund → issue_refund`; the final result is `REFUND: INR 128.25 issued.` Checkpoints are stored under `state/`; run results are in `outputs/agent_runs.jsonl` and the action ledger is `outputs/refund_ledger.jsonl`.

**ReAct versus plan-and-execute (under 250 words):** ReAct asks the model after each observation, so the measured S1 run uses five model calls (882 estimated tokens), S5 uses three (328 tokens), and S6 uses two (66 tokens). A plan-and-execute design could request a complete tool sequence in one call, usually reducing calls and prompt/response overhead when the order data matches expectations. However, a preplanned sequence is less responsive when a tool returns an unexpected result: for example, `get_order` can return NOT_FOUND or an order can be cancelled, and the agent must stop instead of calculating or issuing a refund. ReAct can inspect that observation and choose a safe next step; the explicit iteration, token, and time limits also provide a clear human handoff. For refunds, I would choose bounded ReAct because correctness and avoiding an unauthorised refund are more important than saving a few model calls. The deterministic mock is only a simulation, so these counts describe this exercise rather than a production LLM.

# Limitations and Next Steps

- All records and model responses are synthetic. The offline mock is deterministic keyword/policy logic, not a real language model; performance does not imply production quality.
- The refund ledger records intended actions in a local JSONL file and does not move money. A real workflow would require authentication, authorisation, idempotency, audit controls and human approval for high-value cases.
- The retention list is illustrative and not a leakage-free evaluation because its Premium flag is derived from the whole-window RFM table, including the September period used to define the lapse label. The model features themselves use only pre-2025-09-01 orders.
- Next steps include monitoring model drift, adding stronger schema checks and tests, documenting operational ownership, and validating routing and refund rules with authorised business stakeholders before any real deployment.
