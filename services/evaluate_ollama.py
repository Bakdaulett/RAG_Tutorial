import sys
import time
from pathlib import Path

import ollama
import pandas as pd

from embedding_manager import Embedder
from qdrant_manager import QdrantManager
from llm_judge import LLMJudge


def load_questions_from_excel(excel_path: Path, question_col: str = "C", answer_col: str = "D"):
    """
    Load question/answer pairs from Excel.

    Columns are specified by letter (e.g. 'C' for questions, 'D' for answers).
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)

    q_idx = ord(question_col.upper()) - ord("A")
    a_idx = ord(answer_col.upper()) - ord("A")

    questions = df.iloc[:, q_idx].dropna().tolist()
    answers = df.iloc[:, a_idx].dropna().tolist()

    n = min(len(questions), len(answers))
    questions = questions[:n]
    answers = answers[:n]

    return questions, answers


class LocalRAGEvaluator:
    """
    Evaluate RAG using only local components (Ollama + Qdrant) to avoid Gemini quotas.
    - Retrieval: Embedder + QdrantManager
    - Generation: Ollama (e.g. llama2)
    - Judge: LLMJudge (Ollama)
    """

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        embedding_model: str = "nomic-embed-text",
        top_k: int = 5,
        score_threshold: float = 0.5,
        generation_model: str = "llama3.1:8b-instruct-q4_0",
        judge_model: str = "llama3.1:8b-instruct-q4_0",
        results_dir: Path | None = None,
    ):
        self.collection_name = collection_name
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.generation_model = generation_model
        self.embedder = Embedder(model_name=embedding_model)
        self.qdrant_manager = QdrantManager()
        self.judge = LLMJudge(model_name=judge_model)

        if results_dir is None:
            results_dir = Path(__file__).resolve().parent / "results"
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results: list[dict] = []

    def retrieve_contexts(self, query: str) -> list[str]:
        """Retrieve relevant contexts from Qdrant."""
        print(f"Embedding query: {query[:60]}...")
        query_embedding = self.embedder.embed_text(query)

        print(f"Searching Qdrant (top_k={self.top_k}, threshold={self.score_threshold})...")
        try:
            results = self.qdrant_manager.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=self.top_k,
                score_threshold=self.score_threshold,
            )

            hits = results.points if hasattr(results, "points") else results

            processed = [
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                }
                for hit in hits
            ]
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            processed = []

        contexts = [r["text"] for r in processed]

        print(f"Retrieved {len(contexts)} contexts")
        for i, r in enumerate(processed, 1):
            print(
                f"  Context {i}: score={r['score']:.3f}, "
                f"source={r['metadata'].get('source', 'unknown')}"
            )

        return contexts

    def generate_with_rag(self, query: str, contexts: list[str]) -> str:
        """Generate an answer using Ollama with retrieved contexts."""
        if not contexts:
            print("No contexts found, generating without RAG (local Ollama)...")
            prompt = (
                "You are a helpful AI assistant. Answer the user's question directly.\n\n"
                f"User Question: {query}\n\nAnswer:"
            )
        else:
            contexts_text = "\n\n".join(
                [f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)]
            )
            prompt = (
                "You are a helpful AI assistant. Answer the user's question based on the "
                "provided context documents.\n\n"
                f"Context Documents:\n{contexts_text}\n\n"
                f"User Question: {query}\n\n"
                "Instructions:\n"
                "- Answer based primarily on the information in the context documents.\n"
                "- If the context doesn't contain enough information, say so clearly.\n"
                "- Be concise and accurate.\n"
                "- If relevant, mention which context(s) you used.\n\n"
                "Answer:"
            )

        print("Generating answer with local Ollama model...")
        response = ollama.chat(
            model=self.generation_model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.get("message", {}).get("content", "").strip()

    def process_one(self, question: str, true_answer: str) -> dict:
        """Process a single QA pair: retrieve, generate, judge."""
        from datetime import datetime

        start = time.time()

        contexts = self.retrieve_contexts(question)
        answer = self.generate_with_rag(question, contexts)
        judgment = self.judge.evaluate(answer, true_answer, query=question)

        elapsed = time.time() - start

        result = {
            "timestamp": datetime.now().isoformat(),
            "query": question,
            "response": answer,
            "true_response": true_answer,
            "contexts": contexts,
            "num_contexts": len(contexts),
            "judgment": judgment,
            "elapsed_time": elapsed,
        }

        self.results.append(result)
        return result

    def save_text_summary(self):
        """Save a simple text summary: Right answer: X/Y, accuracy: Z%."""
        from datetime import datetime

        total = len(self.results)
        judged = [r for r in self.results if r["judgment"] is not None]
        correct = sum(1 for r in judged if r["judgment"].get("judgment"))

        accuracy = (correct / len(judged) * 100) if judged else 0.0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"evaluation_result_{timestamp}.txt"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(
                f"Right answer: {correct}/{len(judged)}, "
                f"accuracy: {accuracy:.0f}%"
            )

        print(f"\nSimple accuracy summary saved to: {summary_file}")


def run_excel_evaluation():
    """
    Evaluate all question–answer pairs from RAG Documents.xlsx using local RAG + judge.
    """
    print("\n" + "=" * 80)
    print("RAG EXCEL EVALUATION (LOCAL OLLAMA + QDRANT)")
    print("=" * 80)

    project_root = Path(__file__).resolve().parent.parent
    excel_path = project_root / "RAG Documents.xlsx"

    print(f"\nLoading questions from: {excel_path}")
    questions, answers = load_questions_from_excel(
        excel_path, question_col="C", answer_col="D"
    )
    total = len(questions)
    print(f"Loaded {total} question–answer pairs.\n")

    evaluator = LocalRAGEvaluator(collection_name="pdf_documents")

    start_time = time.time()

    for i, (q, true_ans) in enumerate(zip(questions, answers), start=1):
        print("\n" + "-" * 80)
        print(f"Question {i}/{total}")
        print("-" * 80)
        print(f"Q: {q}")

        result = evaluator.process_one(q, true_ans)

        print("\nModel answer:")
        print(result["response"])

        j = result["judgment"]
        status = "CORRECT" if j.get("judgment") else "INCORRECT"
        print(
            f"\nJudge: {status} (confidence: {j.get('confidence', 0.0):.2f})"
        )
        print(f"Explanation: {j.get('explanation', '')}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed/60:.1f} minutes")

    evaluator.save_text_summary()


if __name__ == "__main__":
    try:
        run_excel_evaluation()
        sys.exit(0)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        sys.exit(1)

