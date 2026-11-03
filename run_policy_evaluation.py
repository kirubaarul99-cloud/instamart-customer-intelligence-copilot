"""Compare overlapped and non-overlapped policy chunks on the supplied evaluation set."""
import json
from pathlib import Path
import pandas as pd
from src.policy_rag import PolicyRAG, evaluate
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
rows=[]
for label, overlap in [("overlap_10",10),("overlap_0",0)]:
    rag=PolicyRAG(chunk_size=40,overlap=overlap)
    df=evaluate(rag)
    answerable=df[df.expected_doc!="NONE"]
    unanswerable=df[df.expected_doc=="NONE"]
    summary={"configuration":label,"chunk_count":len(rag.chunks),
      "expected_doc_retrieved":f"{int(answerable.retrieved_doc.sum())}/{len(answerable)}",
      "expected_fact_found":f"{int(answerable.fact_found.sum())}/{len(answerable)}",
      "fallback_correct":f"{int(unanswerable.fallback.sum())}/{len(unanswerable)}",
      "failed_questions":df.loc[(df.expected_doc!="NONE") & (~df.fact_found),"question"].tolist()}
    if overlap == 10:
        assert summary["expected_doc_retrieved"] == "12/12" and summary["expected_fact_found"] == "12/12" and summary["fallback_correct"] == "2/2"
    else:
        assert summary["expected_fact_found"] == "10/12" and summary["fallback_correct"] == "2/2"
    rows.append(summary)
    print(summary)
(OUT/"policy_eval_summary.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
