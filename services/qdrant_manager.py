import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantManager:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        """
        Initialize QdrantManager with local Qdrant connection.

        Args:
            qdrant_url: URL for local Qdrant instance
        """
        self.client = QdrantClient(url=qdrant_url)

    def create_collection(self, name: str, vector_size: int = 768):
        """
        Create a new collection in Qdrant.

        Args:
            name: Name of the collection to create
            vector_size: Dimension of vectors (768 for nomic-embed-text)
        """
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Collection '{name}' created successfully")
        except Exception as e:
            print(f"Error creating collection: {e}")

    def insert_point(self, embedding: List[float], collection_name: str, chunk_text: str,
                     metadata: Optional[dict] = None):
        """
        Insert a point (embedding + text) into a Qdrant collection.

        Args:
            embedding: The embedding vector
            collection_name: Name of the collection
            chunk_text: The original text chunk
            metadata: Optional additional metadata to store
        """
        point_id = str(uuid.uuid4())

        payload = {
            "text": chunk_text,
            **(metadata or {})
        }

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            print(f"Point inserted with ID: {point_id}")
        except Exception as e:
            print(f"Error inserting point: {e}")

    def insert_points_batch(self, embeddings: List[List[float]], collection_name: str, chunk_texts: List[str],
                            metadata_list: Optional[List[dict]] = None):
        """
        Insert multiple points in batch for better performance.

        Args:
            embeddings: List of embedding vectors
            collection_name: Name of the collection
            chunk_texts: List of text chunks
            metadata_list: Optional list of metadata dicts
        """
        if len(embeddings) != len(chunk_texts):
            raise ValueError("Number of embeddings must match number of texts")

        points = []
        for i, (embedding, text) in enumerate(zip(embeddings, chunk_texts)):
            payload = {
                "text": text,
                **(metadata_list[i] if metadata_list and i < len(metadata_list) else {})
            }

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
            points.append(point)

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Batch inserted {len(points)} points")
        except Exception as e:
            print(f"Error inserting batch: {e}")

    def search_point(self, embedding: List[float], collection_name: str, top_k: int = 5,
                     score_threshold: float = 0.75) -> List[str]:
        """
        Search for similar points in the collection.

        Args:
            embedding: Query embedding vector
            collection_name: Name of the collection to search
            top_k: Number of top results to return
            score_threshold: Minimum similarity score (0-1 for cosine)

        Returns:
            List of text chunks that match the query
        """
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=embedding,
                limit=top_k,
                score_threshold=score_threshold
            )

            # Extract text from results
            texts = [hit.payload.get("text", "") for hit in results.points]
            return texts
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def search_by_text(self, query_text: str, collection_name: str, embedding_function, top_k: int = 5,
                       score_threshold: float = 0.75) -> List[dict]:
        """
        Search using text query (automatically converts to embedding).

        Args:
            query_text: Text query to search for
            collection_name: Name of the collection to search
            embedding_function: Function to generate embeddings (from Embedder)
            top_k: Number of top results to return
            score_threshold: Minimum similarity score

        Returns:
            List of dicts containing text and score
        """
        # Get embedding for query text
        query_embedding = embedding_function(query_text)

        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )

            # Return text and scores
            return [
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                }
                for hit in results.points
            ]
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def delete_collection(self, name: str):
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name=name)
            print(f"Collection '{name}' deleted successfully")
        except Exception as e:
            print(f"Error deleting collection: {e}")

    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []

    def get_collection_info(self, name: str):
        """Get information about a collection."""
        try:
            return self.client.get_collection(collection_name=name)
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None