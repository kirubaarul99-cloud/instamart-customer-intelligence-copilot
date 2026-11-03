"""Run the seven required refund scenarios plus safe-stop and resume demonstrations."""
from pathlib import Path
import csv, json
from src.refund_agent import run_agent, resume_agent, OUT, STATE

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT.mkdir(exist_ok=True)
STATE.mkdir(exist_ok=True)
# Keep the committed run artifacts reproducible and avoid stale ledger entries.
for name in ("agent_runs.jsonl", "refund_ledger.jsonl"):
    p = OUT / name
    if p.exists(): p.unlink()
for p in STATE.glob("*.json"):
    p.unlink()

with (RAW / "agent_scenarios.csv").open(newline="", encoding="utf-8") as f:
    scenarios = list(csv.DictReader(f))
results = []
with (OUT / "agent_runs.jsonl").open("w", encoding="utf-8") as f:
    for row in scenarios:
        result = run_agent(row["message"], scenario_id=row["scenario_id"])
        item = {"scenario_id": row["scenario_id"], **result,
                "refund_amount": next((s["observation"].get("refund_inr") for s in result["steps"]
                                        if s["tool"] in ("issue_refund", "escalate_to_human")), 0.0),
                "expected_outcome": row["expected_outcome"],
                "expected_refund_inr": float(row["expected_refund_inr"])}
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        assert item["status"] == ("FALLBACK_MAX_ITERATIONS" if row["scenario_id"] == "S7" else row["expected_outcome"]), item
        assert abs(float(item["refund_amount"]) - float(row["expected_refund_inr"])) <= 0.01, item
        results.append(item)

# Required token-budget demonstration.
s1 = scenarios[0]
token_demo = run_agent(s1["message"], scenario_id="S1-token-demo", token_limit=200)
# Required deterministic timeout demonstration: no model call should be made.
time_demo = run_agent(s1["message"], scenario_id="S1-time-demo", time_limit_seconds=0)

# Required crash/resume demonstration: crash after the second tool has been checkpointed.
resume_run_id = "resume-S1"
try:
    run_agent(s1["message"], scenario_id="S1-resume", run_id=resume_run_id, crash_after=2)
except RuntimeError as exc:
    if "checkpoint" not in str(exc): raise
resumed = resume_agent(resume_run_id)
checkpoint = json.loads((STATE / f"{resume_run_id}.json").read_text(encoding="utf-8"))
# The checkpoint records the full ordered tool history, including the two pre-crash calls.
resume_tools = checkpoint["tools_executed"]
assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
assert time_demo["status"] == "FALLBACK_TIMEOUT" and time_demo["llm_calls"] == 0
assert resumed["status"] == "REFUND" and "128.25" in resumed["final_answer"]
assert resume_tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"]

print("SCENARIOS")
for r in results:
    print(r["scenario_id"], r["status"], r["refund_amount"], r["llm_calls"], r["tokens_used"],
          "expected", r["expected_outcome"], r["expected_refund_inr"])
print("TOKEN_DEMO", token_demo["status"], token_demo["llm_calls"], token_demo["tokens_used"])
print("TIME_DEMO", time_demo["status"], time_demo["llm_calls"], time_demo["tokens_used"])
print("RESUME", resumed["status"], resumed["final_answer"], resume_tools, resumed["llm_calls"], resumed["tokens_used"])
