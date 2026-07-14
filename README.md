# Junior AI Engineer — Technical Assessment

**Format:** Take-home (target 2–3 hours, hard cap 24h to submit) + 45min live debrief
**AI tools:** Allowed and expected. Use Claude, Copilot, ChatGPT, whatever you normally use. You will be asked to explain and extend your own code live, without AI assistance, in the debrief — so don't submit anything you can't defend.

**Submission:**

1. Clone the repo we sent you (a copy of this template, unique to you).
2. Commit and push your work directly to it as you go — don't submit one giant commit.
3. Your code, `README.md` decisions, and the written answers from Part D and E should all be committed there — **no zip files**, we want to see your commit history.
4. When done, let us know you've pushed your final commit.

---

## Getting started

```bash
git clone <this-repo-url>
cd jr_ai_engineer_test
pip install -r requirements.txt
```

Everything referenced below (`train_churn.py`, `llm_client_stub.py`, `Dockerfile`, `data/customers.csv`) lives at the root of this repo — there's no separate starter-code folder, it's all yours to edit in place.

---

## Context (fictional but realistic)

You work at a mid-size e-commerce company. The support team is drowning in tickets. Two problems have landed on your desk:

1. **Churn**: the retention team wants to know, per customer, whether they're likely to cancel next month.
2. **Ticket triage**: support tickets come in as free text and need to be auto-tagged by category before routing.

You'll build a small, real, working slice of both.

---

## Part A — Classic ML build (design + build)

Dataset: `data/customers.csv` (synthetic, provided — ~5,000 rows, churn label included, deliberately a bit messy: missing values, one leaky column, one mislabeled dtype, mild class imbalance).

**Task:**

1. EDA — identify and document the data quality issues (yes, including the leaky column — tell us why it's leaky, don't just drop it silently).
2. Build a churn classifier. Any classical approach is fine (logistic regression, gradient boosting, etc.) — we're not grading algorithm choice, we're grading whether your reasoning for the choice holds up.
3. Evaluate properly: pick a metric appropriate to the class balance, justify it in the README, report it on a held-out set.
4. Serialize the trained model (`model.pkl` or equivalent) so it can be loaded by Part C.

**Starter code:** `train_churn.py` — intentionally has one bug (a silent data leak from train/test split done in the wrong order). Find it and fix it. Don't just rewrite the file from scratch with an LLM; the debrief will ask you to point to the exact line and explain why it mattered.

## Part A Implementation Notes

I dropped `days_since_last_cancellation_request` because it is a leaky feature. In this dataset, `-1` means the customer did not make a cancellation request, while non-negative values are strongly tied to customers who churned. That gives the model information that is too close to the answer. The model can learn to predict churn based on it, which is not available at prediction time.

I also fixed the order of preprocessing. I changed the order to split first, then calculate missing-value medians based only on the training split and propagate those same median values to the test split. I also fit the scaler only on the training data. The test set is only transformed using values learned from the training set.

The classes are about 66% non-churned and 34% churned, so accuracy is misleading because a majority-class predictor can already get about 66% accuracy.
I report ROC-AUC as the primary metric because the retention team would use ranked churn probabilities to decide which customers to contact first. I also report churn-class precision and recall because missing a likely churner is more costly than sending an unneeded retention offer.

---

## Part B — GenAI build (design + build)

**Task:** Build a ticket-classification pipeline that calls an LLM (use any provider; a mock/stub client is provided if you don't have API keys — `llm_client_stub.py`) to classify a support ticket into one of: `billing`, `shipping`, `technical`, `account`, `other`.

Requirements:

- Structured output (JSON) parsed reliably — handle the case where the model returns malformed JSON or an invalid category. Don't let one bad response crash the pipeline.
- A documented prompt (system + user template) in your repo, not just inline.
- At least 5 hand-written test cases (`tests/test_classifier.py`), including one adversarial/edge case (e.g. a ticket that plausibly fits two categories, or is empty).
- One paragraph in the README: how would you evaluate this classifier's accuracy at scale without hand-labeling thousands of tickets?

We are explicitly testing whether you understand that LLM output is unreliable by default and needs to be engineered around, not whether you can write a clever prompt.

## Part B Notes

To evaluate this classifier at scale without any hand-labeling, I would run human spot-checks on a small random sample (in the order of 150 tickets ) to estimate ground-truth accuracy with confidence intervals. This is enough to benchmark different LLM providers or prompt versions. Over time, I would collect implicit feedback: when a human agent manually re-categorizes a ticket, I log that correction as a label. This builds a labeled dataset organically without ever asking anyone to label tickets just for evaluation.

---

## Part C — Deploy (build + design)

**Task:** Wrap the Part A model behind a `/predict` endpoint using FastAPI.

- `POST /predict` takes a customer's features as JSON, returns churn probability + predicted label.
- Input validation (Pydantic) — reject malformed input with a clear 4xx, don't 500.
- `Dockerfile` provided at the repo root — it's broken (won't build). Fix it. Note what was wrong in your README.
- One test hitting the endpoint (unit or integration, your call).

We're not expecting production-grade infra here — no Kubernetes, no CI pipeline required for the take-home. We're checking: does it run, does it fail gracefully, can you containerize something correctly.

---

## Part D — MLOps & ML judgment (written, no code required)

Answer in your README, ~150–250 words each. We want your actual reasoning, not textbook definitions.

1. Three months after deploying the churn model, prediction quality has quietly degraded. Walk through how you'd detect that this happened, and what you'd check first to diagnose _why_.
2. What's the difference between monitoring a classic ML model in production and monitoring an LLM-based feature? Name one metric specific to each that the other doesn't need.
3. Your model.pkl from Part A was trained on data up to today. In 3 months you'll retrain. What needs to be versioned besides the model file itself, and why does it matter if you skip it?

## Part D Notes

### 1. Detecting and Diagnosing Model Degradation

I would monitor two things.

First, I would compare production feature distributions with the training data every week. Large changes in features such as `monthly_charge`, `tenure_months`, `support_tickets`, `age`, or `contract_type` could indicate data drift.

Second, once true churn labels become available, I would compare predicted probabilities with actual churn outcomes.

To diagnose the cause, I would first check the data pipeline. I would verify that preprocessing, feature names, data types, missing-value handling, and encoding are still the same as during training. Then, I would check whether customer behavior has changed.

### 2. Monitoring Classic ML vs. an LLM

A classic ML model produces a fixed output, such as a churn probability. Its performance depends on production data remaining similar to the training data. I would monitor feature drift, prediction distributions, and model performance when labels become available.

An LLM generates text, so I would also monitor whether its response makes sense (no hallucination), follows instructions, and respects the expected format.

A classic ML-specific metric is **feature drift**, such as changes in the distributions of `monthly_charge` or `support_tickets`.

An LLM-specific metric is **malformed response rate**, for example the percentage of outputs that do not follow the required JSON schema. The churn model does not need this metric because it always returns a numeric probability.

### 3. What to Version Besides `model.pkl`

I would version everything needed to reproduce the training process.

This includes the exact training data snapshot, the training and preprocessing code, the Git commit hash, the feature list, model parameters, random seed, evaluation results, and exact dependency versions in `requirements.txt`.

This matters because data, code, and libraries can change over time. Without these versions, I may not be able to reproduce the same model, compare retraining results fairly, understand why performance changed.

---

## Part E — Business framing (written, no code required)

The support team lead tells you: _"Just have the AI answer the tickets automatically, we don't need a human at all."_

Write a half-page response (in the README) covering:

- What you'd actually recommend building first, and why it's not full automation on day one.
- One success metric you'd propose to the business (not a model metric — a business one), and how you'd measure it.
- One concrete failure mode of full automation here, and what it would cost the company if it went wrong.

This is graded on business judgment and communication to a non-technical stakeholder, not on ML knowledge.

---

## What we are NOT grading

- Whether you used AI tools to write code (you should).
- Polish/UI — nothing here is user-facing.
- Algorithm sophistication — a logistic regression that's correctly evaluated beats an XGBoost with a data leak.

## What we ARE grading

- Can you spot and explain bugs/leaks in code you didn't write.
- Do you understand _why_ your evaluation approach is right for this data, not just how to call `.fit()`.
- Do you treat LLM output as unreliable and defensively engineer around it.
- Can you translate a technical decision into a sentence a support team lead would accept.
- In the live debrief: is the reasoning actually yours.
