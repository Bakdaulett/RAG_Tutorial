from typing import Literal
import json
from pydantic_ai import Agent


class RouterAgent:
    """
    Router Agent that decides whether a query needs RAG or can be answered directly.
    Uses Gemini to analyze the query and make intelligent routing decisions.
    """

    def __init__(self, model, document_themes: list[str] = None):
        """
        Initialize Router Agent.

        Args:
            model: Gemini model instance from pydantic_ai GoogleModel
            document_themes: List of themes/topics available in the document collection
        """
        self.model = model
        self.agent = Agent(model=model)
        self.document_themes = document_themes or [
            "machine learning",
            "artificial intelligence",
            "deep learning",
            "neural networks",
            "data science",
            "computer vision",
            "natural language processing",
            "reinforcement learning"
        ]

    def route(self, query: str) -> dict:
        """
        Analyze query and decide routing.

        Args:
        Args:
            query: User's question

        Returns:
            dict with 'decision' ('rag' or 'direct') and 'reasoning'
        """

        prompt = f"""You are a routing agent for a RAG (Retrieval-Augmented Generation) system.

Available document themes in the knowledge base:
{', '.join(self.document_themes)}

User Query: "{query}"

Analyze this query and decide:
- Use "rag" if the query asks about technical concepts, specific information, or topics that would benefit from document retrieval
- Use "direct" if the query is:
  * A greeting or casual conversation
  * A general knowledge question not requiring specific documents
  * A creative task (write a poem, story, etc.)
  * A question about current events or common knowledge

Respond in JSON format:
{{
    "decision": "rag" or "direct",
    "reasoning": "brief explanation of your decision"
}}

Only output valid JSON, nothing else."""

        try:
            response = self.agent.run_sync(prompt)

            # Parse JSON from response (pydantic-ai returns result as string directly)
            response_text = str(response.output).strip() if hasattr(response, 'output') else str(response).strip()

            # Extract JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)

            # Validate decision
            if result.get("decision") not in ["rag", "direct"]:
                print(f"Invalid decision from router, defaulting to RAG")
                return {"decision": "rag", "reasoning": "Default fallback"}

            return result

        except Exception as e:
            print(f"Error in router agent: {e}")
            # Default to RAG on error
            return {
                "decision": "rag",
                "reasoning": f"Error in routing, defaulting to RAG: {str(e)}"
            }

    def should_use_rag(self, query: str) -> bool:
        """
        Simple boolean check if RAG should be used.

        Args:
            query: User's question

        Returns:
            True if should use RAG, False otherwise
        """
        result = self.route(query)
        return result["decision"] == "rag"