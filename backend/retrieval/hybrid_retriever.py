"""Hybrid retrieval combining BM25 and semantic search with RRF."""
from typing import List, Dict, Tuple
import numpy as np
from retrieval.bm25_retriever import bm25_retriever
from vector_store.faiss_store import vector_store

class HybridRetriever:
    """Combine BM25 sparse and dense semantic retrieval using Reciprocal Rank Fusion."""
    
    def __init__(self):
        """Initialize hybrid retriever."""
        self.k_rrf = 60  # RRF parameter
    
    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[Dict, float]],
        semantic_results: List[Tuple[Dict, float]]
    ) -> List[Tuple[Dict, float]]:
        """
        Apply Reciprocal Rank Fusion to combine rankings.
        
        RRF score = sum(1 / (k + rank)) for each ranking system
        """
        # Create movie ID to data mapping
        movie_scores = {}
        
        # Process BM25 results
        for rank, (movie, score) in enumerate(bm25_results, start=1):
            movie_id = movie.get("id")
            if movie_id:
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = {
                        "movie": movie,
                        "rrf_score": 0,
                        "bm25_score": score,
                        "semantic_score": 0
                    }
                movie_scores[movie_id]["rrf_score"] += 1 / (self.k_rrf + rank)
        
        # Process semantic results
        for rank, (movie, score) in enumerate(semantic_results, start=1):
            movie_id = movie.get("id")
            if movie_id:
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = {
                        "movie": movie,
                        "rrf_score": 0,
                        "bm25_score": 0,
                        "semantic_score": score
                    }
                else:
                    movie_scores[movie_id]["semantic_score"] = score
                movie_scores[movie_id]["rrf_score"] += 1 / (self.k_rrf + rank)
        
        # Sort by RRF score
        sorted_results = sorted(
            movie_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        # Return as list of (movie, rrf_score) tuples
        return [(item["movie"], item["rrf_score"]) for item in sorted_results]
    
    def search(self, query: str, k: int = 20) -> List[Tuple[Dict, float]]:
        """
        Perform hybrid search combining BM25 and semantic retrieval.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (movie, score) tuples sorted by RRF score
        """
        # Retrieve from both systems
        bm25_results = bm25_retriever.search(query, k=k*2)
        semantic_results = vector_store.search(query, k=k*2)
        
        # Apply RRF
        fused_results = self.reciprocal_rank_fusion(bm25_results, semantic_results)
        
        return fused_results[:k]
    
    def get_retrieval_stats(self, query: str, k: int = 20) -> Dict:
        """Get statistics about the retrieval process."""
        bm25_results = bm25_retriever.search(query, k=k)
        semantic_results = vector_store.search(query, k=k)
        fused_results = self.reciprocal_rank_fusion(bm25_results, semantic_results)
        
        bm25_ids = {m.get("id") for m, _ in bm25_results}
        semantic_ids = {m.get("id") for m, _ in semantic_results}
        
        return {
            "bm25_count": len(bm25_results),
            "semantic_count": len(semantic_results),
            "overlap": len(bm25_ids & semantic_ids),
            "unique_total": len(bm25_ids | semantic_ids),
            "fused_count": len(fused_results)
        }

# Global hybrid retriever instance
hybrid_retriever = HybridRetriever()
