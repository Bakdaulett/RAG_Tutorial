import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from main import RAGSystem


def load_api_keys_and_models_from_env() -> list[tuple[str, list[str]]]:
    """
    Load Gemini API keys and their associated models from .env file.
    
    Expected format in .env:
        GEMINI_API_KEY_1=your-key-1
        GEMINI_MODELS_1=gemini-2.5-flash-lite,gemini-2.5-flash
        GEMINI_API_KEY_2=your-key-2
        GEMINI_MODELS_2=gemini-2.5-pro
        ...
    
    Returns list of tuples: [(api_key, [model1, model2, ...]), ...]
    """
    load_dotenv()
    
    api_configs = []
    for i in range(1, 7):  # Keys 1-6
        key_name = f"GEMINI_API_KEY_{i}"
        models_name = f"GEMINI_MODELS_{i}"
        
        api_key = os.getenv(key_name)
        models_str = os.getenv(models_name)
        
        if api_key:
            # Parse models (comma-separated) or use default
            if models_str:
                models = [m.strip() for m in models_str.split(",") if m.strip()]
            else:
                # Default model if none specified
                models = ["gemini-2.5-flash-lite"]
            
            api_configs.append((api_key.strip(), models))
            print(f"✓ API Key {i}: {len(models)} model(s) - {', '.join(models)}")
        else:
            print(f"⚠️  Warning: {key_name} not found in .env file")
    
    if not api_configs:
        raise RuntimeError(
            "No Gemini API keys found in .env file!\n"
            "Please add GEMINI_API_KEY_1 (and optionally GEMINI_MODELS_1) to your .env file.\n"
            "Example:\n"
            "  GEMINI_API_KEY_1=your-key-here\n"
            "  GEMINI_MODELS_1=gemini-2.5-flash-lite,gemini-2.5-flash"
        )
    
    print(f"\n✓ Loaded {len(api_configs)} API key configuration(s)")
    return api_configs


COLLECTION_NAME = "pdf_documents"
CHECKPOINT_FILE = Path(__file__).resolve().parent / "results" / "gemini_eval_checkpoint.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_questions_from_excel(excel_path: Path, question_col: str = "C", answer_col: str = "D"):
    """Load question/answer pairs from Excel."""
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


def load_checkpoint() -> tuple[int, list[dict]]:
    """Load checkpoint: returns (last_processed_index, accumulated_results)."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("last_index", 0), data.get("results", [])
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return 0, []
    return 0, []


def save_checkpoint(last_index: int, results: list[dict]):
    """Save checkpoint with current progress."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_index": last_index, "results": results}, f, indent=2)
    print(f"\n✓ Checkpoint saved: processed {last_index} questions")


def save_intermediate_results(results: list[dict], api_key_index: int, model_name: str = ""):
    """Save intermediate results when switching API keys or models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = f"_{model_name.replace('.', '_').replace('-', '_')}" if model_name else ""
    
    # Save detailed CSV
    csv_file = RESULTS_DIR / f"gemini_results_api{api_key_index}{model_suffix}_{timestamp}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    import csv as csv_module
    fieldnames = [
        "timestamp", "query", "routing_decision", "routing_reasoning",
        "response", "num_contexts", "true_response",
        "judgment_judgment", "judgment_confidence", "judgment_explanation",
        "elapsed_time",
    ]
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            judgment = r.get("judgment") or {}
            writer.writerow({
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
            })
    
    print(f"✓ Intermediate results saved to: {csv_file}")


def is_quota_error(error: Exception) -> bool:
    """Check if error is a quota/limit error."""
    error_str = str(error).lower()
    return (
        "429" in error_str
        or "quota" in error_str
        or "resource_exhausted" in error_str
        or "api key expired" in error_str
        or "api_key_invalid" in error_str
        or "limit" in error_str
    )


def calculate_and_save_final_stats(all_results: list[dict]):
    """Calculate final statistics and save TXT summary."""
    total = len(all_results)
    rag_count = sum(1 for r in all_results if r.get("routing", {}).get("decision") == "rag")
    direct_count = total - rag_count

    judged_results = [r for r in all_results if r.get("judgment") is not None]
    correct_count = sum(1 for r in judged_results if r["judgment"].get("judgment"))

    accuracy = (correct_count / len(judged_results) * 100) if judged_results else 0.0

    # Save TXT summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = RESULTS_DIR / f"gemini_evaluation_final_{timestamp}.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Right answer: {correct_count}/{len(judged_results)}, accuracy: {accuracy:.0f}%")

    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"Total questions processed: {total}")
    print(f"RAG decisions: {rag_count}, Direct decisions: {direct_count}")
    print(f"Judged: {len(judged_results)}")
    print(f"Correct: {correct_count}, Incorrect: {len(judged_results) - correct_count}")
    print(f"Accuracy: {accuracy:.0f}%")
    print(f"\nFinal summary saved to: {summary_file}")


def run_gemini_evaluation():
    """
    Evaluate all questions using Gemini with API key and model rotation.
    
    - Uses RAGSystem (Gemini for routing + generation)
    - Rotates through API keys and models from .env file when hitting quota limits
    - For each API key, tries all associated models before moving to next key
    - Saves checkpoint after each question
    - Saves intermediate results when switching keys/models
    - Continues from last processed question
    """
    print("\n" + "=" * 80)
    print("GEMINI EXCEL EVALUATION (WITH API KEY & MODEL ROTATION)")
    print("=" * 80)

    # Load API keys and models from .env file
    try:
        api_configs = load_api_keys_and_models_from_env()
    except RuntimeError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

    # Load questions
    project_root = Path(__file__).resolve().parent.parent
    excel_path = project_root / "RAG Documents.xlsx"

    print(f"\nLoading questions from: {excel_path}")
    questions, answers = load_questions_from_excel(excel_path, question_col="C", answer_col="D")
    total = len(questions)
    print(f"Loaded {total} question–answer pairs.\n")

    # Load checkpoint (resume from last position)
    start_index, accumulated_results = load_checkpoint()
    if start_index > 0:
        print(f"📌 Resuming from question {start_index + 1}/{total}")
        print(f"   Already processed: {len(accumulated_results)} questions\n")
    else:
        print("🆕 Starting fresh evaluation\n")

    # Initialize with first API key and model
    current_key_index = 0
    current_model_index = 0
    rag_system: Optional[RAGSystem] = None
    all_results = accumulated_results.copy()

    start_time = time.time()

    try:
        while start_index < total and current_key_index < len(api_configs):
            api_key, models = api_configs[current_key_index]
            
            # Try each model for this API key
            model_exhausted = False
            
            while current_model_index < len(models) and start_index < total:
                model_name = models[current_model_index]
                
                print(f"\n{'=' * 80}")
                print(f"API Key {current_key_index + 1}/{len(api_configs)}, Model {current_model_index + 1}/{len(models)}")
                print(f"Model: {model_name}")
                print(f"{'=' * 80}")

                # Initialize RAG system with current API key and model
                try:
                    rag_system = RAGSystem(
                        api_key=api_key,
                        collection_name=COLLECTION_NAME,
                        model_name=model_name,
                        results_dir=str(RESULTS_DIR),
                    )
                except Exception as e:
                    print(f"❌ Failed to initialize with API key {current_key_index + 1}, model {model_name}: {e}")
                    current_model_index += 1
                    continue

                # Process questions until quota limit or completion
                questions_processed_this_config = 0

                for i in range(start_index, total):
                    q = questions[i]
                    true_ans = answers[i]

                    print("\n" + "-" * 80)
                    print(f"Question {i + 1}/{total}")
                    print("-" * 80)
                    print(f"Q: {q}")

                    try:
                        # Process query (this may use 2 Gemini calls: router + generator)
                        result = rag_system.process_query(q, true_response=true_ans, save_result=True)

                        print("\nModel answer:")
                        print(result["response"][:200] + "..." if len(result["response"]) > 200 else result["response"])

                        if result.get("judgment"):
                            j = result["judgment"]
                            status = "CORRECT" if j.get("judgment") else "INCORRECT"
                            print(f"\nJudge: {status} (confidence: {j.get('confidence', 0.0):.2f})")

                        # Add to accumulated results
                        all_results.append(result)
                        questions_processed_this_config += 1

                        # Save checkpoint after each question
                        save_checkpoint(i + 1, all_results)

                    except Exception as e:
                        if is_quota_error(e):
                            print(f"\n⚠️  Quota limit reached with API key {current_key_index + 1}, model {model_name}")
                            print(f"   Error: {str(e)[:200]}...")

                            # Save intermediate results for this API key + model
                            if questions_processed_this_config > 0:
                                new_results = all_results[-questions_processed_this_config:]
                                save_intermediate_results(new_results, current_key_index + 1, model_name)

                            # Try next model for this API key
                            current_model_index += 1
                            start_index = i  # Continue from this question
                            model_exhausted = True
                            break
                        else:
                            # Non-quota error: log and continue
                            print(f"\n⚠️  Error processing question {i + 1}: {e}")
                            # Still save checkpoint
                            save_checkpoint(i + 1, all_results)
                            continue

                # If we completed all questions, break
                if i >= total - 1:
                    print(f"\n✅ Completed all {total} questions!")
                    break

                # If model was exhausted, break inner loop to try next model
                if model_exhausted:
                    break

                # If we processed all questions with this model, move to next model
                if start_index >= total:
                    break

            # If all models for this API key are exhausted, move to next API key
            if current_model_index >= len(models):
                print(f"\n⚠️  All models exhausted for API key {current_key_index + 1}. Moving to next API key.")
                current_key_index += 1
                current_model_index = 0  # Reset model index for next API key
            elif start_index >= total:
                # All questions completed
                break

        # If we ran out of API keys
        if current_key_index >= len(api_configs):
            print(f"\n⚠️  All API keys and models exhausted. Processed {len(all_results)}/{total} questions.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving progress...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Final save
        elapsed = time.time() - start_time
        print(f"\n{'=' * 80}")
        print("EVALUATION SESSION COMPLETE")
        print(f"{'=' * 80}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Questions processed: {len(all_results)}/{total}")

        if all_results:
            save_checkpoint(len(all_results), all_results)
            calculate_and_save_final_stats(all_results)
        else:
            print("No results to save.")


if __name__ == "__main__":
    try:
        run_gemini_evaluation()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
