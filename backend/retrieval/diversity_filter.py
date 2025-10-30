"""Maximal Marginal Relevance (MMR) for diversity filtering."""
from typing import List, Dict, Tuple
import numpy as np
from vector_store.embeddings import embedding_manager

class DiversityFilter:
    """Apply MMR to ensure diverse recommendations."""
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize diversity filter.
        
        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        self.lambda_param = lambda_param
    
    def apply_mmr(
        self,
        candidates: List[Tuple[Dict, float]],
        query: str = None,
        k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        Apply Maximal Marginal Relevance for diversity.
        
        MMR = λ * relevance - (1-λ) * max_similarity_to_selected
        
        Args:
            candidates: List of (movie, relevance_score) tuples
            query: Optional query for relevance calculation
            k: Number of diverse results to return
        
        Returns:
            Diverse subset of candidates
        """
        if len(candidates) <= k:
            return candidates
        
        # Extract embeddings
        embeddings = []
        for movie, score in candidates:
            if "embedding" in movie:
                emb = np.array(movie["embedding"])
            else:
                emb = embedding_manager.generate_movie_embedding(movie)
                movie["embedding"] = emb.tolist()
            
            # Normalize
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Initialize
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        # Select first item (highest relevance)
        first_idx = 0
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select diverse items
        for _ in range(k - 1):
            if not remaining_indices:
                break
            
            max_mmr_score = -float('inf')
            max_mmr_idx = None
            
            for idx in remaining_indices:
                # Relevance score
                relevance = candidates[idx][1]
                
                # Calculate max similarity to already selected items
                max_sim = max(
                    self._cosine_similarity(embeddings[idx], embeddings[sel_idx])
                    for sel_idx in selected_indices
                )
                
                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                
                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    max_mmr_idx = idx
            
            if max_mmr_idx is not None:
                selected_indices.append(max_mmr_idx)
                remaining_indices.remove(max_mmr_idx)
        
        # Return selected candidates
        return [candidates[idx] for idx in selected_indices]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(vec1, vec2))
    
    def filter_by_genre_diversity(
        self,
        candidates: List[Tuple[Dict, float]],
        max_per_genre: int = 3
    ) -> List[Tuple[Dict, float]]:
        """Ensure genre diversity by limiting movies per genre."""
        genre_counts = {}
        filtered = []
        
        for movie, score in candidates:
            genres = movie.get("genres", [])
            genre_names = [
                g.get("name", g) if isinstance(g, dict) else str(g)
                for g in genres
            ]
            
            # Check if any genre is over-represented
            can_add = True
            for genre in genre_names:
                if genre_counts.get(genre, 0) >= max_per_genre:
                    can_add = False
                    break
            
            if can_add:
                filtered.append((movie, score))
                for genre in genre_names:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        return filtered

# Global diversity filter instance
diversity_filter = DiversityFilter(lambda_param=0.7)
