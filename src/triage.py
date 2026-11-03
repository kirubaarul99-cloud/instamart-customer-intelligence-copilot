"""Part C: privacy-aware ticket classification with retries, validation and routing."""
from pathlib import Path
import json, random, re, time
import pandas as pd
import mock_llm

PROJECT = Path(__file__).resolve().parents[1]
RAW_DIR, OUTPUT_DIR = PROJECT / "data/raw", PROJECT / "outputs"
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED = {"billing", "delivery", "technical", "fraud"}

ZERO = ('CLASSIFY. Allowed categories: billing, delivery, technical, fraud. '
        'Return JSON only in the form {"category":"...","reason":"..."}.')
FEW = ZERO + '''
EXAMPLES:
Text: Unauthorized transaction, I suspect fraud. -> {"category":"fraud","reason":"unauthorized transaction"}
Text: My order arrived late. -> {"category":"delivery","reason":"late delivery"}
Text: App crashes at login. -> {"category":"technical","reason":"app crash"}
'''
PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


def redact(text):
    totals = {}
    safe = text
    for label, pattern, placeholder in PII_PATTERNS:
        safe, count = pattern.subn(placeholder, safe)
        totals[label] = count
    return safe, totals


def _call_with_retries(messages, ticket_id):
    for retry_number in range(4):  # initial call + at most three retries
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_number == 3:
                raise
            delay = BASE_DELAY * (2 ** retry_number) * random.uniform(0.9, 1.15)
            print(f"retry ticket={ticket_id} retry={retry_number} wait={delay:.4f}s")
            time.sleep(delay)


def _category_from_reply(text):
    try:
        payload = json.loads(text)
        category = payload.get("category") if isinstance(payload, dict) else None
        if category in ALLOWED:
            return category, "classified_json"
        return "", "flagged_invalid_output"
    except (json.JSONDecodeError, TypeError, AttributeError):
        # Hyphenated strings such as billing-related are split with the rubric's regex.
        found = {word for word in re.findall(r"[a-z]+", (text or "").lower()) if word in ALLOWED}
        if len(found) == 1:
            return next(iter(found)), "classified_keyword"
        return "", "flagged_invalid_output"


def run_pipeline(prompt=FEW, token_budget=100000):
    tickets = pd.read_csv(RAW_DIR / "tickets.csv", keep_default_na=False)
    segments_path = OUTPUT_DIR / "customer_segments.csv"
    segments = pd.read_csv(segments_path) if segments_path.exists() else pd.DataFrame(
        columns=["customer_id", "segment_name", "is_premium"])
    if not segments.empty:
        segment_lookup = segments.set_index("customer_id").to_dict("index")
    else:
        segment_lookup = {}
    mock_llm.reset()
    random.seed(42)
    spent_tokens = 0
    redaction_counts = {name: 0 for name, _, _ in PII_PATTERNS}
    output_rows = []

    for ticket in tickets.itertuples(index=False):
        text = str(ticket.text)
        category, status, used = "", "", 0
        if not text.strip():
            status = "skipped_empty"
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        else:
            safe_text, counts = redact(text)
            for key, value in counts.items():
                redaction_counts[key] += value
            for _, pattern, _ in PII_PATTERNS:
                assert pattern.search(safe_text) is None, "PII pattern remained after redaction"
            if spent_tokens >= token_budget:
                status = "budget_exhausted"
            else:
                # The user content is precisely the redacted ticket, with no extra wrapper.
                response = _call_with_retries([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": safe_text},
                ], ticket.ticket_id)
                used = response.usage.total_tokens
                spent_tokens += used
                category, status = _category_from_reply(response.choices[0].message.content or "")

        segment = segment_lookup.get(ticket.customer_id, {})
        if status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif bool(segment.get("is_premium", False)):
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
        output_rows.append({"ticket_id": ticket.ticket_id, "customer_id": ticket.customer_id,
                            "status": status, "category": category,
                            "segment_name": segment.get("segment_name", ""),
                            "priority": priority, "route": route, "tokens_used": used})

    result = pd.DataFrame(output_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = "triage_results.csv" if token_budget >= 100000 and prompt == FEW else "triage_budget_demo.csv"
    result.to_csv(OUTPUT_DIR / target, index=False)
    return result, {"tokens": spent_tokens, "redactions": redaction_counts,
                    "status_counts": result.status.value_counts().to_dict()}
