"""Part D: bounded, resumable refund agent using the supplied offline mock LLM."""
from pathlib import Path
from datetime import datetime, timezone
import json, sqlite3, time, uuid
import mock_llm
from src.policy_rag import PolicyRAG

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"
rag = PolicyRAG()


def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id=?",
            (order_id,),
        ).fetchone()
    if not row:
        return {"error": "order not found"}
    return dict(zip(("order_id", "status", "order_amount", "delivery_minutes"), row))


def search_policy(query):
    return {"results": rag.retrieve(query, 3)}


def calculate_refund(order_amount, delivery_minutes):
    amount = float(order_amount) if delivery_minutes is not None and float(delivery_minutes) > 45 else 0.0
    return {"refund_inr": round(amount, 2)}


def _ledger(tool, order_id, refund_inr):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "refund_ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}
TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the clean SQLite database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve relevant passages from the policy documents.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Calculate the eligible late-delivery refund amount.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["number", "null"]}}, "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Record an approved refund in the local audit ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Record a high-value refund for human approval.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
]


def _status(content):
    return (content or "").split(":", 1)[0].strip() or "UNKNOWN"


def _fallback(status, steps, llm_calls, tokens_used):
    return {"status": status, "final_answer": f"{status}: hand off to a human agent; review the completed steps before taking action.",
            "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}


def _checkpoint(run_id, scenario_id, messages, tokens_used, step_number, tools_executed, steps, llm_calls, user_message):
    STATE.mkdir(parents=True, exist_ok=True)
    obj = {"run_id": run_id, "scenario_id": scenario_id, "user_message": user_message,
           "messages": messages, "tokens_used": tokens_used, "step_number": step_number,
           "tools_executed": tools_executed, "steps": steps, "llm_calls": llm_calls}
    (STATE / f"{run_id}.json").write_text(json.dumps(obj, default=str), encoding="utf-8")


def _continue(state, max_iterations, token_limit, time_limit_seconds, crash_after=None):
    run_id = state["run_id"]
    scenario_id = state["scenario_id"]
    messages = state["messages"]
    tokens_used = state["tokens_used"]
    steps = state["steps"]
    tools_executed = state["tools_executed"]
    llm_calls = state["llm_calls"]
    start = time.monotonic()
    # max_iterations is the maximum total number of model calls for this run, including calls before resume.
    while llm_calls < max_iterations:
        if time.monotonic() - start >= time_limit_seconds:
            return _fallback("FALLBACK_TIMEOUT", steps, llm_calls, tokens_used)
        response = mock_llm.chat(messages)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        if tokens_used > token_limit:
            return _fallback("FALLBACK_TOKEN_LIMIT", steps, llm_calls, tokens_used)
        msg = response.choices[0].message
        if not msg.tool_calls:
            content = msg.content or ""
            return {"status": _status(content), "final_answer": content, "steps": steps,
                    "llm_calls": llm_calls, "tokens_used": tokens_used}
        tc = msg.tool_calls[0]
        args = tc.arguments
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": args}]})
        if tc.name not in TOOL_MAP:
            observation = {"error": f"unknown tool: {tc.name}"}
        else:
            observation = TOOL_MAP[tc.name](**args)
        tools_executed.append(tc.name)
        steps.append({"tool": tc.name, "arguments": args, "observation": observation})
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                         "content": json.dumps(observation)})
        _checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), tools_executed, steps, llm_calls,
                    next((m["content"] for m in messages if m["role"] == "user"), ""))
        if crash_after is not None and len(tools_executed) == crash_after:
            raise RuntimeError("Simulated crash after checkpoint was saved")
    return _fallback("FALLBACK_MAX_ITERATIONS", steps, llm_calls, tokens_used)


def run_agent(user_message, scenario_id="demo", max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after=None):
    run_id = run_id or str(uuid.uuid4())
    state = {"run_id": run_id, "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": "AGENT: verify orders and follow refund policy."},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "step_number": 0, "tools_executed": [], "steps": [], "llm_calls": 0}
    _checkpoint(run_id, scenario_id, state["messages"], 0, 0, [], [], 0, user_message)
    return _continue(state, max_iterations, token_limit, time_limit_seconds, crash_after)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    path = STATE / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found for run_id={run_id}")
    state = json.loads(path.read_text(encoding="utf-8"))
    return _continue(state, max_iterations, token_limit, time_limit_seconds)
