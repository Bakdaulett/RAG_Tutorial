import json
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from router_agent import RouterAgent
from rag_generator import RAGGenerator
from llm_judge import LLMJudge
from dotenv import load_dotenv


class RAGSystem:
    """
    Complete RAG system with Router Agent and LLM Judge.

    Workflow:
    1. User Query → Router Agent (decide RAG or Direct)
    2. RAG Generator (if RAG) or Direct LLM (if Direct)
    3. LLM Judge evaluates response
    4. Save results and statistics
    """

    def __init__(
            self,
            api_key: str,
            collection_name: str,
            model_name: str,
            results_dir: str = str(Path(__file__).resolve().parent / "results")
    ):
        """
        Initialize RAG System.

        Args:
            api_key: Google API key for Gemini
            collection_name: Qdrant collection name
            model_name: Gemini model name
            results_dir: Directory to save results
        """
        print("INITIALIZING RAG SYSTEM")
        print("—" * 80)

        # Initialize Gemini
        print("\n[1/4] Setting up Gemini...")
        self.provider = GoogleProvider(api_key=api_key)
        self.model = GoogleModel(model_name, provider=self.provider)

        # Initialize Router Agent
        print("\n[2/4] Setting up Router Agent...")
        self.router = RouterAgent(self.model)

        # Initialize RAG Generator
        print("\n[3/4] Setting up RAG Generator...")
        self.rag_generator = RAGGenerator(
            model=self.model,
            collection_name=collection_name
        )

        # Initialize LLM Judge (Ollama-based, separate from Gemini)
        print("\n[4/4] Setting up LLM Judge (Ollama)...")
        self.judge = LLMJudge()

        # Results tracking
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_log = []

        print("\n" + "RAG SYSTEM IS READY ✓")
        print("—" * 80 + "\n")
        print()

    def process_query(
            self,
            query: str,
            true_response: Optional[str] = None,
            save_result: bool = True
    ) -> dict:
        """
        Process a single query through the complete pipeline.

        Args:
            query: User's question
            true_response: Optional ground truth for evaluation
            save_result: Whether to save the result

        Returns:
            dict with complete result information
        """
        print("\n" + "PROCESSING QUERY")
        print("—" * 80)
        print(f"Query: {query}")

        start_time = time.time()

        # Step 1: Router decides RAG or Direct
        print("\n[STEP 1] Router Agent Decision...")
        routing_decision = self.router.route(query)
        print(f"Decision: {routing_decision['decision'].upper()}")
        print(f"Reasoning: {routing_decision['reasoning']}")

        # Step 2: Generate response based on routing
        print(f"\n[STEP 2] Generating Response ({routing_decision['decision']})...")

        if routing_decision['decision'] == 'rag':
            # Use RAG pipeline
            rag_result = self.rag_generator.rag_based_answer(query)
            response = rag_result['answer']
            contexts = rag_result['contexts']
            num_contexts = rag_result['num_contexts']
        else:
            # Use direct LLM
            response = self.rag_generator.generate_direct(query)
            contexts = []
            num_contexts = 0

        print(f"Response generated: {response[:100]}...")

        # Step 3: LLM Judge evaluation (if true response provided)
        judgment_result = None
        if true_response:
            print("\n[STEP 3] LLM Judge Evaluation...")
            judgment_result = self.judge.evaluate(response, true_response, query)
            print(f"Judgment: {judgment_result['judgment']}")
            print(f"Confidence: {judgment_result['confidence']:.2f}")
            print(f"Explanation: {judgment_result['explanation']}")
        else:
            print("\n[STEP 3] Skipped (no true response provided)")

        elapsed_time = time.time() - start_time

        # Build result object
        result = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "routing": routing_decision,
            "response": response,
            "contexts": contexts,
            "num_contexts": num_contexts,
            "true_response": true_response,
            "judgment": judgment_result,
            "elapsed_time": elapsed_time
        }

        # Save result
        if save_result:
            self.results_log.append(result)

        print(f"\n[COMPLETE] Processed in {elapsed_time:.2f} seconds")
        print("—" * 80 + "\n")

        return result

    def batch_process(
            self,
            queries: list[dict],
            save_stats: bool = True
    ) -> dict:
        """
        Process multiple queries and generate statistics.

        Args:
            queries: List of dicts with 'query' and optional 'true_response'
            save_stats: Whether to save statistics

        Returns:
            dict with statistics
        """
        print("\n" + "BATCH PROCESSING")
        print("—" * 80)
        print(f"Total queries: {len(queries)}\n")

        results = []

        for i, item in enumerate(queries, 1):
            print(f"\n{'—' * 80}")
            print(f"Query {i}/{len(queries)}")
            print(f"{'—' * 80}")

            query = item['query']
            true_response = item.get('true_response')

            result = self.process_query(query, true_response)
            results.append(result)

        # Calculate statistics
        stats = self.calculate_statistics(results)

        if save_stats:
            self.save_statistics(stats)
            self.save_detailed_results(results)

        return stats

    def calculate_statistics(self, results: list[dict]) -> dict:
        """
        Calculate statistics from results.

        Args:
            results: List of result dicts

        Returns:
            Statistics dict
        """
        total = len(results)
        rag_count = sum(1 for r in results if r['routing']['decision'] == 'rag')
        direct_count = total - rag_count

        # Judgment statistics (only for queries with true responses)
        judged_results = [r for r in results if r['judgment'] is not None]
        correct_count = sum(1 for r in judged_results if r['judgment']['judgment'])

        avg_time = sum(r['elapsed_time'] for r in results) / total if total > 0 else 0
        avg_contexts = sum(r['num_contexts'] for r in results) / total if total > 0 else 0

        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": total,
            "routing": {
                "rag": rag_count,
                "direct": direct_count,
                "rag_percentage": (rag_count / total * 100) if total > 0 else 0
            },
            "judgment": {
                "total_judged": len(judged_results),
                "correct": correct_count,
                "incorrect": len(judged_results) - correct_count,
                "accuracy": (correct_count / len(judged_results) * 100) if judged_results else 0
            },
            "performance": {
                "avg_time_seconds": avg_time,
                "avg_contexts_retrieved": avg_contexts
            }
        }

        print("\n" + "STATISTICS")
        print("=" * 80)
        print(json.dumps(stats, indent=2))

        return stats

    def save_statistics(self, stats: dict):
        """Save statistics to CSV file."""
        stats_file = self.results_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Flatten nested statistics structure for CSV
        flat_stats = {
            "timestamp": stats.get("timestamp"),
            "total_queries": stats.get("total_queries"),
            "routing_rag": stats.get("routing", {}).get("rag"),
            "routing_direct": stats.get("routing", {}).get("direct"),
            "routing_rag_percentage": stats.get("routing", {}).get("rag_percentage"),
            "judgment_total_judged": stats.get("judgment", {}).get("total_judged"),
            "judgment_correct": stats.get("judgment", {}).get("correct"),
            "judgment_incorrect": stats.get("judgment", {}).get("incorrect"),
            "judgment_accuracy": stats.get("judgment", {}).get("accuracy"),
            "performance_avg_time_seconds": stats.get("performance", {}).get("avg_time_seconds"),
            "performance_avg_contexts_retrieved": stats.get("performance", {}).get("avg_contexts_retrieved"),
        }

        fieldnames = list(flat_stats.keys())

        with open(stats_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(flat_stats)

        print(f"\nStatistics saved to (CSV): {stats_file}")

    def save_detailed_results(self, results: list[dict]):
        """Save detailed results to CSV file."""
        results_file = self.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Define CSV columns
        fieldnames = [
            "timestamp",
            "query",
            "routing_decision",
            "routing_reasoning",
            "response",
            "num_contexts",
            "true_response",
            "judgment_judgment",
            "judgment_confidence",
            "judgment_explanation",
            "elapsed_time",
        ]

        with open(results_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                judgment = r.get("judgment") or {}
                row = {
                    "timestamp": r.get("timestamp"),
                    "query": r.get("query"),
                    "routing_decision": r.get("routing", {}).get("decision"),
                    "routing_reasoning": r.get("routing", {}).get("reasoning"),
                    "response": r.get("response"),
                    "num_contexts": r.get("num_contexts"),
                    "true_response": r.get("true_response"),
                    "judgment_judgment": judgment.get("judgment"),
                    "judgment_confidence": judgment.get("confidence"),
                    "judgment_explanation": judgment.get("explanation"),
                    "elapsed_time": r.get("elapsed_time"),
                }
                writer.writerow(row)

        print(f"Detailed results saved to (CSV): {results_file}")

    def save_text_accuracy_summary(self, stats: dict):
        """
        Save a simple text summary like:
        "Right answer: 51/60, accuracy: 85%"

        Only counted over results that have a judgment.
        """
        judged = stats.get("judgment", {})
        total_judged = judged.get("total_judged", 0)
        correct = judged.get("correct", 0)
        accuracy = judged.get("accuracy", 0.0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"evaluation_summary_{timestamp}.txt"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Right answer: {correct}/{total_judged}, accuracy: {accuracy:.0f}%")

        print(f"Simple text summary saved to: {summary_file}")

def main():
    """
    Main entry point: simple chat loop.
    - Uses RouterAgent (Gemini) to decide RAG vs direct
    - Answers with RAGGenerator accordingly
    - Tracks basic statistics and saves them on exit
    """

    # Load configuration
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    collection_name = os.getenv("PDF_DOCUMENTS")
    model_name = os.getenv("GEMINI_MODEL_NAME")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment or .env file.")

    if not collection_name:
        collection_name = "pdf_documents"

    if not model_name:
        model_name = "gemini-2.5-flash-lite"

    # Initialize system
    rag_system = RAGSystem(
        api_key=api_key,
        collection_name=collection_name,
        model_name=model_name,
    )

    print("RAG CHAT")
    print("—" * 80)
    print("Type your question and press Enter.")
    print("Type 'stats' to see current statistics.")
    print("Type 'quit' or 'exit' to end the chat.")
    print("—" * 80)

    try:
        while True:
            query = input("\nYou: ").strip()

            if not query:
                continue

            lower_q = query.lower()
            if lower_q in {"quit", "exit", "q"}:
                print("Exiting chat...")
                break

            if lower_q == "stats":
                if rag_system.results_log:
                    rag_system.calculate_statistics(rag_system.results_log)
                else:
                    print("No queries yet, nothing to show.")
                continue

            # Process query through router + RAG/direct (no ground truth yet)
            result = rag_system.process_query(query, save_result=True)

            print("Assistant:")
            print(result["response"])

            # Optional: LLM-as-judge evaluation if user provides a reference answer
            true_resp = input(
                "If you have a reference/true answer for this question, "
                "paste it here for LLM judge (or press Enter to skip):\n> "
            ).strip()

            if true_resp:
                judgment = rag_system.judge.evaluate(
                    llm_response=result["response"],
                    true_response=true_resp,
                    query=query,
                )
                # Attach judgment and true answer to the stored result
                result["true_response"] = true_resp
                result["judgment"] = judgment

                print(
                    f"\nLLM Judge: {'CORRECT' if judgment['judgment'] else 'INCORRECT'} "
                    f"(confidence: {judgment.get('confidence', 0):.2f})"
                )
                print(f"Explanation: {judgment.get('explanation', '')}")

    except KeyboardInterrupt:
        print("\n\nInterrupted, exiting chat...")

    # On exit, save statistics, detailed results and simple text accuracy summary
    if rag_system.results_log:
        stats = rag_system.calculate_statistics(rag_system.results_log)
        rag_system.save_statistics(stats)
        rag_system.save_detailed_results(rag_system.results_log)
        rag_system.save_text_accuracy_summary(stats)
        print("\nSession statistics, detailed results, and accuracy summary saved to the 'results' folder.")
    else:
        print("\nNo queries were processed. Nothing to save.")


if __name__ == "__main__":
    main()