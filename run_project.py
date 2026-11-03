"""Run the connected data, customer-modelling, triage, and policy-retrieval stages."""
from src.data_pipeline import audit, clean_data, validate
from src.customer_models import rfm_and_segments, build_lapse_model
from src.triage import run_pipeline, FEW
from src.policy_rag import PolicyRAG, evaluate


def main():
    print("\n[PART A] Raw audit, cleaning, SQLite load, and validation")
    audit()
    clean_data()
    validate()

    print("\n[PART B] RFM segments and lapse prediction")
    _, k_scores, segment_profile, pca_variance = rfm_and_segments(k=4)
    print("K-means diagnostics:\n", k_scores.to_string(index=False))
    print("Segment profile:\n", segment_profile.to_string(index=False))
    print("PCA explained-variance ratios:", pca_variance)
    lapse_results = build_lapse_model()
    print("Lapse cohort:", {key: lapse_results[key] for key in ("cohort_rows", "lapsed", "train_rows", "test_rows", "baseline")})
    print("Model test metrics:", lapse_results["models"])
    print("Training-only balance and feature diagnostics:", lapse_results["diagnostics"])

    print("\n[PART C] Few-shot support-ticket triage")
    triage_rows, triage_report = run_pipeline(prompt=FEW, token_budget=100000)
    print("Triage report:", triage_report)
    print("Sample output:\n", triage_rows.head().to_string(index=False))

    print("\n[PART D] Policy retrieval and grounded answers")
    policy_index = PolicyRAG(chunk_size=40, overlap=10)
    evaluation = evaluate(policy_index)
    answerable = evaluation[evaluation.expected_doc != "NONE"]
    unanswerable = evaluation[evaluation.expected_doc == "NONE"]
    print("Chunk count:", len(policy_index.chunks))
    print("Answerable questions: expected document", int(answerable.retrieved_doc.sum()), "/", len(answerable),
          "; expected fact", int(answerable.fact_found.sum()), "/", len(answerable))
    print("Unanswerable questions correctly fell back:", int(unanswerable.fallback.sum()), "/", len(unanswerable))
    print("Core pipeline complete. Run run_sql_analysis.py, run_triage_experiments.py, run_policy_evaluation.py, and run_agent_scenarios.py to produce the remaining evaluation artifacts.")


if __name__ == "__main__":
    main()
