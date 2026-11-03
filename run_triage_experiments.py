"""Run zero-shot, budget-limited, and final few-shot ticket triage experiments."""
from pathlib import Path
import pandas as pd
from src.triage import run_pipeline, ZERO, FEW
ROOT = Path(__file__).resolve().parent
raw = pd.read_csv(ROOT / "data/raw/tickets.csv", keep_default_na=False)

def report(label, result, summary):
    joined = result.merge(raw[["ticket_id", "true_category"]], on="ticket_id", how="left")
    classified = joined[joined["status"].isin(["classified_json", "classified_keyword"])]
    accuracy = (classified["category"] == classified["true_category"]).mean() if len(classified) else 0.0
    tokens = int(summary["tokens"])
    print(label, {"classified": len(classified), "accuracy": round(float(accuracy), 4),
                  "tokens": tokens, "tokens_per_classified": round(tokens / max(1, len(classified)), 2),
                  "status_counts": summary["status_counts"], "redactions": summary["redactions"]})
    if label == "ZERO_SHOT":
        print("ZERO_SHOT_CONFUSION")
        print(pd.crosstab(classified["true_category"], classified["category"]).to_string())
    return {"classified": len(classified), "accuracy": float(accuracy), "tokens": tokens,
            "tokens_per_classified": tokens / max(1, len(classified)), "status_counts": summary["status_counts"]}

zero, zsum = run_pipeline(prompt=ZERO, token_budget=100000)
z = report("ZERO_SHOT", zero, zsum)
assert z["classified"] == 147 and abs(z["accuracy"] - 126/147) < 1e-9
# Use the zero-shot prompt for the required budget demo. It writes outputs/triage_budget_demo.csv.
budget, bsum = run_pipeline(prompt=ZERO, token_budget=6000)
budget_exhausted = int((budget.status == "budget_exhausted").sum())
assert budget_exhausted > 0
print("BUDGET_DEMO", {"budget": 6000, "budget_exhausted": budget_exhausted,
                       "status_counts": bsum["status_counts"], "tokens": bsum["tokens"]})
# Final deliverable is the few-shot run with a large budget.
few, fsum = run_pipeline(prompt=FEW, token_budget=100000)
f = report("FEW_SHOT", few, fsum)
assert f["classified"] == 147 and f["accuracy"] == 1.0
assert fsum["status_counts"].get("flagged_invalid_output") == 9
