from typing import List
import sys

sys.path.append('/mnt/user-data/uploads')

from embedding_manager import Embedder
from qdrant_manager import QdrantManager
from pydantic_ai import Agent


class RAGGenerator:
    """
    RAG Generator that handles the complete RAG pipeline:
    1. Embed query
    2. Search Qdrant for relevant contexts
    3. Generate answer using LLM + contexts
    """

    def __init__(
            self,
            model,
            collection_name: str = "pdf_documents",
            embedding_model: str = "nomic-embed-text",
            top_k: int = 5,
            score_threshold: float = 0.5
    ):
        """
        Initialize RAG Generator.

        Args:
            model: Gemini model instance from pydantic_ai for generation
            collection_name: Qdrant collection name
            embedding_model: Embedding model name
            top_k: Number of contexts to retrieve
            score_threshold: Minimum similarity score for retrieval
        """
        self.model = model
        self.agent = Agent(model=model)
        self.collection_name = collection_name
        self.top_k = top_k
        self.score_threshold = score_threshold

        # Initialize embedder and Qdrant manager
        self.embedder = Embedder(model_name=embedding_model)
        self.qdrant_manager = QdrantManager()

        print(f"RAG Generator initialized with collection: {collection_name}")

    def retrieve_contexts(self, query: str) -> List[str]:
        """
        Retrieve relevant contexts from Qdrant.

        Args:
            query: User's query text

        Returns:
            List of relevant text chunks
        """
        # Step 1: Embed the query
        print(f"Embedding query: {query[:50]}...")
        query_embedding = self.embedder.embed_text(query)

        # Step 2: Search Qdrant using the embedding directly
        print(f"Searching Qdrant (top_k={self.top_k}, threshold={self.score_threshold})...")
        try:
            # Use the newer Qdrant API (v1.7+)
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            results = self.qdrant_manager.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=self.top_k,
                score_threshold=self.score_threshold
            )

            # Extract points from results
            if hasattr(results, 'points'):
                hits = results.points
            else:
                hits = results

            # Convert to same format as search_by_text
            results = [
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                }
                for hit in hits
            ]
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            import traceback
            traceback.print_exc()
            results = []

        # Step 3: Extract text contexts
        contexts = [result['text'] for result in results]

        print(f"Retrieved {len(contexts)} contexts")
        for i, result in enumerate(results, 1):
            print(f"  Context {i}: score={result['score']:.3f}, "
                  f"source={result['metadata'].get('source', 'unknown')}")

        return contexts

    def generate(self, query: str, contexts: List[str]) -> str:
        """
        Generate answer using LLM with retrieved contexts.

        Args:
            query: User's query
            contexts: List of relevant context chunks

        Returns:
            Generated answer
        """
        if not contexts:
            print("No contexts found, generating without RAG...")
            return self.generate_direct(query)

        # Build prompt with contexts
        contexts_text = "\n\n".join([
            f"[Context {i + 1}]\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context documents.

Context Documents:
{contexts_text}

User Question: {query}

Instructions:
- Answer based primarily on the information in the context documents
- If the context doesn't contain enough information, say so clearly
- Be concise and accurate
- Cite which context(s) you used if relevant

Answer:"""

        print("Generating answer with RAG...")
        response = self.agent.run_sync(prompt)

        return str(response.output).strip() if hasattr(response, 'output') else str(response).strip()

    def generate_direct(self, query: str) -> str:
        """
        Generate answer directly without RAG (no context retrieval).

        Args:
            query: User's query

        Returns:
            Generated answer
        """
        prompt = f"""You are a helpful AI assistant. Answer the user's question directly.

User Question: {query}

Answer:"""

        print("Generating direct answer (no RAG)...")
        response = self.agent.run_sync(prompt)

        return str(response.output).strip() if hasattr(response, 'output') else str(response).strip()

    def rag_based_answer(self, query: str) -> dict:
        """
        Complete RAG pipeline: retrieve contexts and generate answer.

        Args:
            query: User's query

        Returns:
            dict with 'answer', 'contexts', and 'metadata'
        """
        # Retrieve contexts
        contexts = self.retrieve_contexts(query)

        # Generate answer
        answer = self.generate(query, contexts)

        return {
            "answer": answer,
            "contexts": contexts,
            "num_contexts": len(contexts)
        }