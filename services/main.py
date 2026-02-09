import json
from os import getenv
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
            collection_name: str = "pdf_documents",
            model_name: str = "gemini-2.5-flash-lite",
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

        # Initialize LLM Judge
        print("\n[4/4] Setting up LLM Judge...")
        self.judge = LLMJudge(self.model)

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
        """Save statistics to file."""
        stats_file = self.results_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_file}")

    def save_detailed_results(self, results: list[dict]):
        """Save detailed results to file."""
        results_file = self.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert to serializable format
        serializable_results = []
        for r in results:
            serializable_results.append({
                "timestamp": r["timestamp"],
                "query": r["query"],
                "routing_decision": r["routing"]["decision"],
                "routing_reasoning": r["routing"]["reasoning"],
                "response": r["response"],
                "num_contexts": r["num_contexts"],
                "true_response": r["true_response"],
                "judgment": r["judgment"],
                "elapsed_time": r["elapsed_time"]
            })

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Detailed results saved to: {results_file}")

    def interactive_mode(self):
        """
        Interactive mode for testing the system.
        """

        print("\n" +"INTERACTIVE MODE")
        print("—" * 80)
        print("Commands:")
        print("  - Type your query to get an answer")
        print("  - Type 'stats' to see current statistics")
        print("  - Type 'quit' or 'exit' to quit")
        print("—" * 80 + "\n")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    break

                if query.lower() == 'stats':
                    if self.results_log:
                        stats = self.calculate_statistics(self.results_log)
                    else:
                        print("No results yet!")
                    continue

                if not query:
                    continue

                # Process query
                result = self.process_query(query, save_result=True)

                print( "\n" + "ANSWER")
                print(f"{'—' * 80}")
                print(result['response'])
                print(f"{'—' * 80}")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """
    Main function demonstrating the RAG system.
    """

    # Configuration
    load_dotenv()
    API_KEY = getenv("GEMINI_API_KEY")
    COLLECTION_NAME = "pdf_documents"
    MODEL_NAME = "gemini-2.5-flash-lite"

    # Initialize system
    rag_system = RAGSystem(
        api_key=API_KEY,
        collection_name=COLLECTION_NAME,
        model_name=MODEL_NAME
    )

    # Example queries for testing
    test_queries = [
        {
            "query": "What is machine learning?",
            "true_response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "query": "Hello, how are you?",
            "true_response": "I'm doing well, thank you for asking!"
        },
        {
            "query": "Explain neural networks",
            "true_response": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information."
        }
    ]

    # Menu
    print("RAG SYSTEM MENU")
    print("—" * 80)
    print("\nOptions:")
    print("1. Process test queries")
    print("2. Interactive mode")
    print("3. Single query test")
    print("4. Exit")
    print("Type 'stats' at any prompt to see current statistics.")

    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()

            if choice.lower() == "stats":
                if rag_system.results_log:
                    rag_system.calculate_statistics(rag_system.results_log)
                else:
                    print("No results yet!")
                continue

            if choice == "1":
                rag_system.batch_process(test_queries)

            elif choice == "2":
                rag_system.interactive_mode()

            elif choice == "3":
                query = input("\nEnter query: ").strip()
                if query.lower() == "stats":
                    if rag_system.results_log:
                        rag_system.calculate_statistics(rag_system.results_log)
                    else:
                        print("No results yet!")
                    continue
                true_resp = input("Enter true response (or press Enter to skip): ").strip()
                true_resp = true_resp if true_resp else None

                result = rag_system.process_query(query, true_resp)

                print("\n" + "RESULT")
                print(f"{'—' * 80}")
                print(f"Response: {result['response']}")
                if result['judgment']:
                    print(f"Judgment: {result['judgment']['judgment']}")
                print(f"{'—' * 80}")

            elif choice == "4":
                print("\nExiting...")
                if rag_system.results_log:
                    stats = rag_system.calculate_statistics(rag_system.results_log)
                    rag_system.save_statistics(stats)
                    rag_system.save_detailed_results(rag_system.results_log)
                break

            else:
                print("Invalid choice")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()