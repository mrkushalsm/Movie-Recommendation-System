"""TMDb API client with enhanced batch operations and caching."""
import requests
import socket
from typing import List, Dict, Optional, Any
from config import config
from utils.cache_manager import cache_manager

# Force IPv4 to avoid IPv6 connection issues
import urllib3.util.connection as urllib3_cn

def allowed_gai_family():
    """Force IPv4 only to avoid IPv6 connection resets."""
    return socket.AF_INET

urllib3_cn.allowed_gai_family = allowed_gai_family

class TMDbClient:
    """Enhanced TMDb API client with intelligent search and batch operations."""
    
    BASE_URL = "https://api.themoviedb.org/3"
    
    def __init__(self):
        """Initialize TMDb client with Bearer token authentication."""
        self.api_key = config.TMDB_API_KEY
        self.access_token = config.TMDB_ACCESS_TOKEN
        self.session = requests.Session()
        
        # Use Bearer token authentication (like the working Colab notebook)
        if self.access_token:
            self.session.headers.update({
                "accept": "application/json",
                "Authorization": f"Bearer {self.access_token}"
            })
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling and retry."""
        params = params or {}
        # Don't add api_key to params when using Bearer token
        if not self.access_token and self.api_key:
            params["api_key"] = self.api_key
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.BASE_URL}{endpoint}", 
                    params=params,
                    timeout=15
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    print(f"Connection error, retrying... ({attempt + 1}/{max_retries})")
                    continue
                else:
                    print(f"TMDb API connection failed after {max_retries} attempts: {e}")
                    return {}
            except requests.RequestException as e:
                print(f"TMDb API error: {e}")
                return {}
        
        return {}
    
    @cache_manager.cached("tmdb_search", ttl=3600)
    def search_movies(self, query: str, page: int = 1, include_adult: bool = False) -> Dict:
        """Search movies by query."""
        return self._make_request("/search/movie", {
            "query": query, 
            "page": page,
            "include_adult": str(include_adult).lower()
        })
    
    @cache_manager.cached("tmdb_discover", ttl=3600)
    def discover_movies(self, **filters) -> Dict:
        """
        Discover movies with filters using /discover endpoint.
        
        Supported filters:
        - with_genres: Genre IDs (comma-separated)
        - primary_release_date.gte/lte: Date range
        - vote_average.gte/lte: Rating range
        - with_people: Person IDs (actors/directors)
        - with_keywords: Keyword IDs
        - sort_by: popularity.desc, vote_average.desc, etc.
        """
        params = {
            "sort_by": filters.get("sort_by", "popularity.desc"),
            "include_adult": "false",
            "include_video": "false",
            "vote_count.gte": config.MIN_RATING_COUNT,
        }
        
        # Add all provided filters
        params.update({k: v for k, v in filters.items() if v is not None})
        
        return self._make_request("/discover/movie", params)
    
    @cache_manager.cached("tmdb_movie_details", ttl=86400)
    def get_movie_details(self, movie_id: int) -> Dict:
        """Get detailed movie information."""
        return self._make_request(
            f"/movie/{movie_id}",
            {"append_to_response": "credits,keywords,similar,watch/providers"}
        )
    
    @cache_manager.cached("tmdb_credits", ttl=86400)
    def get_movie_credits(self, movie_id: int) -> Dict:
        """Get cast and crew for a movie."""
        return self._make_request(f"/movie/{movie_id}/credits")
    
    @cache_manager.cached("tmdb_similar", ttl=3600)
    def get_similar_movies(self, movie_id: int) -> Dict:
        """Get similar movies."""
        return self._make_request(f"/movie/{movie_id}/similar")
    
    @cache_manager.cached("tmdb_keywords", ttl=86400)
    def get_movie_keywords(self, movie_id: int) -> Dict:
        """Get keywords for a movie."""
        return self._make_request(f"/movie/{movie_id}/keywords")
    
    def batch_discover(
        self,
        filters: Dict,
        max_pages: int = 5,
        quality_filter: bool = True
    ) -> List[Dict]:
        """
        Batch discover movies across multiple pages.
        
        Args:
            filters: Discovery filters
            max_pages: Maximum pages to fetch
            quality_filter: Apply quality filtering
        
        Returns:
            List of movie dictionaries
        """
        all_movies = []
        
        for page in range(1, max_pages + 1):
            filters["page"] = page
            result = self.discover_movies(**filters)
            
            movies = result.get("results", [])
            if not movies:
                break
            
            if quality_filter:
                movies = self._apply_quality_filter(movies)
            
            all_movies.extend(movies)
            
            # Stop if we've reached the last page
            if page >= result.get("total_pages", 0):
                break
        
        return all_movies
    
    def _apply_quality_filter(self, movies: List[Dict]) -> List[Dict]:
        """Filter movies by quality criteria."""
        return [
            movie for movie in movies
            if (
                movie.get("vote_count", 0) >= config.MIN_RATING_COUNT
                and movie.get("vote_average", 0) >= config.MIN_VOTE_AVERAGE
                and movie.get("overview")  # Must have overview
                and movie.get("poster_path")  # Must have poster
            )
        ]
    
    def enrich_movie_data(self, movie: Dict) -> Dict:
        """Enrich movie data with additional details."""
        movie_id = movie.get("id")
        if not movie_id:
            return movie
        
        details = self.get_movie_details(movie_id)
        
        # Merge additional data
        movie["full_details"] = details
        movie["genres"] = details.get("genres", [])
        movie["keywords"] = self.get_movie_keywords(movie_id).get("keywords", [])
        movie["credits"] = details.get("credits", {})
        
        return movie
    
    @cache_manager.cached("tmdb_genres", ttl=86400)
    def get_genres(self) -> Dict:
        """Get list of all movie genres."""
        return self._make_request("/genre/movie/list")
    
    def search_person(self, name: str) -> Dict:
        """Search for a person (actor/director)."""
        return self._make_request("/search/person", {"query": name})

# Global client instance
tmdb_client = TMDbClient()
