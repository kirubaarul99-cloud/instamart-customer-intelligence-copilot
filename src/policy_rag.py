"""Part D: TF-IDF retrieval and context-only policy answers."""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

PROJECT = Path(__file__).resolve().parents[1]
POLICY_DIR = PROJECT / "data/policies"


class PolicyRAG:
    def __init__(self, chunk_size=40, overlap=10):
        if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size >= 1 and 0 <= overlap < chunk_size")
        self.chunk_size, self.overlap = chunk_size, overlap
        self.chunks = []
        step = chunk_size - overlap
        for document in sorted(POLICY_DIR.glob("*.txt")):
            words = document.read_text(encoding="utf-8").split()
            for start in range(0, len(words), step):
                passage = words[start:start + chunk_size]
                self.chunks.append({"source": document.name, "text": " ".join(passage)})
                if start + chunk_size >= len(words):
                    break
        if not self.chunks:
            raise FileNotFoundError(f"No policy documents found under {POLICY_DIR}")
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform([chunk["text"] for chunk in self.chunks])

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, self.document_vectors).ravel()
        ranked = similarity.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(similarity[i]),
                 "text": self.chunks[i]["text"]} for i in ranked]

    def answer(self, question):
        passages = self.retrieve(question, 3)
        # Source labels stay on separate lines so the offline model can see provenance.
        context = "\n".join(f"[{item['source']}]\n{item['text']}" for item in passages)
        messages = [
            {"role": "system", "content": "POLICY ASSISTANT: answer only from supplied context. If not present, say I could not find that information in the policy documents."},
            {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
        ]
        response = mock_llm.chat(messages)
        return {"answer": response.choices[0].message.content,
                "sources": list(dict.fromkeys(item["source"] for item in passages)),
                "retrieved": passages}


def evaluate(rag):
    questions = pd.read_csv(PROJECT / "data/raw/policy_eval.csv", keep_default_na=False)
    outcomes = []
    for item in questions.itertuples(index=False):
        result = rag.answer(item.question)
        expected_document = item.expected_doc
        outcomes.append({
            "question": item.question,
            "expected_doc": expected_document,
            "retrieved_doc": expected_document in result["sources"] if expected_document != "NONE" else False,
            "fact_found": item.expected_fact in result["answer"] if item.expected_fact else False,
            "fallback": result["answer"].startswith("I could not find"),
        })
    return pd.DataFrame(outcomes)
