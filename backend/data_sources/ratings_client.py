"""Client for fetching and validating ratings from multiple sources."""
import requests
from typing import Dict, Optional, Tuple
from utils.cache_manager import cache_manager

class RatingsClient:
    """Cross-validate ratings from multiple sources."""
    
    def __init__(self):
        """Initialize ratings client."""
        pass
    
    @cache_manager.cached("ratings_consensus", ttl=86400)
    def get_consensus_rating(
        self,
        movie_title: str,
        year: Optional[int] = None,
        tmdb_rating: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Get consensus rating from multiple sources.
        
        Returns:
            Dict with ratings from different sources and consensus score
        """
        ratings = {}
        
        # TMDb rating (already available)
        if tmdb_rating is not None:
            ratings["tmdb"] = {
                "score": tmdb_rating,
                "scale": 10,
                "normalized": tmdb_rating / 10
            }
        
        # Try to get OMDb (IMDb) rating
        imdb_rating = self._get_omdb_rating(movie_title, year)
        if imdb_rating:
            ratings["imdb"] = imdb_rating
        
        # Calculate consensus
        if ratings:
            consensus = self._calculate_consensus(ratings)
            return {
                "ratings": ratings,
                "consensus_score": consensus,
                "confidence": len(ratings) / 3  # Out of 3 possible sources
            }
        
        return {"ratings": {}, "consensus_score": 0, "confidence": 0}
    
    def _get_omdb_rating(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        Get rating from OMDb API (IMDb data).
        
        Note: Requires OMDb API key. For demo, returns mock data.
        """
        # In production, use actual OMDb API
        # For now, return None to indicate unavailable
        return None
    
    def _get_rt_rating(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        Get Rotten Tomatoes rating.
        
        Note: RT API is restricted. Using web scraping would require rottentomatoes-python
        """
        # In production, implement RT scraping
        return None
    
    def _calculate_consensus(self, ratings: Dict) -> float:
        """Calculate weighted consensus score (0-1 scale)."""
        if not ratings:
            return 0.0
        
        weights = {
            "tmdb": 0.35,
            "imdb": 0.40,
            "rt": 0.25
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for source, rating_data in ratings.items():
            if source in weights:
                weight = weights[source]
                score = rating_data.get("normalized", 0)
                weighted_sum += score * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def validate_rating_quality(self, movie_data: Dict) -> Tuple[bool, str]:
        """
        Validate if a movie has sufficient rating data.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        vote_count = movie_data.get("vote_count", 0)
        vote_average = movie_data.get("vote_average", 0)
        
        if vote_count < 50:
            return False, "Insufficient votes"
        
        if vote_average == 0:
            return False, "No rating available"
        
        return True, "Valid"

# Global client instance
ratings_client = RatingsClient()
