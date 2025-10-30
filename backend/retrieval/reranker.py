"""Re-ranking module for intelligent scoring and filtering."""
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
from config import config
from vector_store.embeddings import embedding_manager

class IntelligentReRanker:
    """Re-rank retrieved movies using composite scoring."""
    
    def __init__(self):
        """Initialize re-ranker with weights from config."""
        self.semantic_weight = config.SEMANTIC_WEIGHT
        self.genre_weight = config.GENRE_WEIGHT
        self.rating_weight = config.RATING_WEIGHT
        self.recency_weight = config.RECENCY_WEIGHT
        self.popularity_weight = config.POPULARITY_WEIGHT
    
    def calculate_composite_score(
        self,
        movie: Dict,
        query: str,
        semantic_similarity: float,
        query_genres: List[str] = None
    ) -> float:
        """
        Calculate composite score for re-ranking.
        
        Score = 0.25*semantic + 0.2*genre + 0.2*rating + 0.15*recency + 0.1*popularity + 0.1*keyword_match
        """
        # Semantic similarity (already calculated)
        semantic_score = semantic_similarity
        
        # Genre overlap
        genre_score = self._calculate_genre_overlap(movie, query_genres)
        
        # Rating score (normalized 0-1)
        rating_score = self._normalize_rating(movie.get("vote_average", 0))
        
        # Recency boost
        recency_score = self._calculate_recency_boost(movie)
        
        # Popularity signal (log-normalized)
        popularity_score = self._normalize_popularity(movie.get("popularity", 0))
        
        # Keyword match bonus (NEW!)
        keyword_match_score = self._calculate_keyword_match(movie, query)
        
        # Weighted sum (adjusted weights to include keyword matching)
        composite = (
            0.25 * semantic_score +      # Reduced from 0.3
            0.2 * genre_score +
            0.2 * rating_score +
            0.15 * recency_score +
            0.1 * popularity_score +
            0.1 * keyword_match_score    # NEW: 10% weight for keyword matching
        )
        
        return composite
    
    def _calculate_keyword_match(self, movie: Dict, query: str) -> float:
        """Calculate keyword matching bonus based on title/overview."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Remove common words
        stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'but', 'me', 'about'}
        query_keywords = query_words - stop_words
        
        if not query_keywords:
            return 0.0
        
        score = 0.0
        
        # Check title
        title = (movie.get("title") or "").lower()
        title_words = set(title.split())
        title_matches = len(query_keywords & title_words)
        score += (title_matches / len(query_keywords)) * 0.6  # Title matches worth 60%
        
        # Check overview
        overview = (movie.get("overview") or "").lower()
        overview_words = set(overview.split())
        overview_matches = len(query_keywords & overview_words)
        score += (overview_matches / len(query_keywords)) * 0.4  # Overview matches worth 40%
        
        return min(score, 1.0)
    
    def _calculate_genre_overlap(self, movie: Dict, query_genres: List[str]) -> float:
        """Calculate genre overlap score."""
        if not query_genres:
            return 0.5  # Neutral score if no genre filter
        
        movie_genres = movie.get("genres", [])
        movie_genre_names = {
            g.get("name", g).lower() if isinstance(g, dict) else str(g).lower()
            for g in movie_genres
        }
        
        query_genre_names = {g.lower() for g in query_genres}
        
        if not movie_genre_names:
            return 0.0
        
        overlap = len(movie_genre_names & query_genre_names)
        max_possible = len(query_genre_names)
        
        return overlap / max_possible if max_possible > 0 else 0.0
    
    def _normalize_rating(self, rating: float) -> float:
        """Normalize rating to 0-1 scale."""
        return min(rating / 10.0, 1.0)
    
    def _calculate_recency_boost(self, movie: Dict) -> float:
        """Calculate recency boost (newer movies get higher scores)."""
        release_date = movie.get("release_date") or movie.get("first_air_date")
        
        if not release_date:
            return 0.0
        
        try:
            if isinstance(release_date, str):
                release_year = int(release_date.split("-")[0])
            else:
                release_year = release_date.year
            
            current_year = datetime.now().year
            age = current_year - release_year
            
            # Exponential decay: score decreases as age increases
            # Movies from current year: 1.0, 5 years ago: ~0.6, 10 years ago: ~0.4
            score = np.exp(-age / 10.0)
            return min(score, 1.0)
        
        except (ValueError, AttributeError):
            return 0.0
    
    def _normalize_popularity(self, popularity: float) -> float:
        """Normalize popularity using log scale."""
        if popularity <= 0:
            return 0.0
        
        # Log normalization (popularity typically ranges 0-1000+)
        # Map log(1) -> 0, log(100) -> ~0.5, log(1000) -> ~0.75
        log_pop = np.log10(popularity + 1)
        normalized = min(log_pop / 3.0, 1.0)  # Divide by 3 to map log10(1000)=3 to 1.0
        
        return normalized
    
    def rerank(
        self,
        results: List[Tuple[Dict, float]],
        query: str,
        query_genres: List[str] = None,
        max_results: int = None
    ) -> List[Tuple[Dict, float]]:
        """
        Re-rank results using composite scoring.
        
        Args:
            results: List of (movie, initial_score) tuples
            query: Original query string
            query_genres: Genre filters from query
            max_results: Maximum number of results to return
        
        Returns:
            Re-ranked list of (movie, composite_score) tuples
        """
        reranked = []
        
        for movie, initial_score in results:
            # Use initial score as semantic similarity
            composite_score = self.calculate_composite_score(
                movie,
                query,
                initial_score,
                query_genres
            )
            
            reranked.append((movie, composite_score))
        
        # Sort by composite score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        if max_results:
            reranked = reranked[:max_results]
        
        return reranked
    
    def get_score_breakdown(self, movie: Dict, query: str, semantic_sim: float) -> Dict:
        """Get detailed score breakdown for explainability."""
        return {
            "semantic_similarity": semantic_sim,
            "genre_overlap": self._calculate_genre_overlap(movie, []),
            "rating_score": self._normalize_rating(movie.get("vote_average", 0)),
            "recency_boost": self._calculate_recency_boost(movie),
            "popularity_score": self._normalize_popularity(movie.get("popularity", 0)),
            "composite_score": self.calculate_composite_score(movie, query, semantic_sim)
        }

# Global re-ranker instance
reranker = IntelligentReRanker()
