"""BM25 sparse retrieval for keyword-based search."""
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import re

class BM25Retriever:
    """BM25-based sparse retrieval for exact keyword matching."""
    
    def __init__(self):
        """Initialize BM25 retriever."""
        self.bm25 = None
        self.movies = []
        self.tokenized_corpus = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def _create_movie_text(self, movie: Dict) -> str:
        """Create searchable text representation of a movie."""
        components = []
        
        # Title
        if movie.get("title"):
            components.append(movie["title"])
        
        # Genres
        genres = movie.get("genres", [])
        if genres:
            genre_names = [g.get("name", g) if isinstance(g, dict) else str(g) for g in genres]
            components.extend(genre_names)
        
        # Keywords
        keywords = movie.get("keywords", [])
        if keywords:
            keyword_names = [k.get("name", k) if isinstance(k, dict) else str(k) for k in keywords]
            components.extend(keyword_names)
        
        # Cast (top 5)
        credits = movie.get("credits", {})
        cast = credits.get("cast", [])[:5]
        for actor in cast:
            if isinstance(actor, dict):
                components.append(actor.get("name", ""))
        
        # Director
        crew = credits.get("crew", [])
        directors = [c.get("name", "") for c in crew if c.get("job") == "Director"]
        components.extend(directors)
        
        # Overview
        if movie.get("overview"):
            components.append(movie["overview"])
        
        return " ".join(components)
    
    def build_index(self, movies: List[Dict]):
        """Build BM25 index from movies."""
        self.movies = movies
        self.tokenized_corpus = []
        
        for movie in movies:
            text = self._create_movie_text(movie)
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            print(f"Built BM25 index with {len(self.movies)} movies")
    
    def search(self, query: str, k: int = 20) -> List[Tuple[Dict, float]]:
        """
        Search using BM25.
        
        Returns:
            List of (movie, score) tuples
        """
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = scores.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                results.append((self.movies[idx], float(scores[idx])))
        
        return results
    
    def add_movies(self, new_movies: List[Dict]):
        """Add new movies to the index."""
        self.movies.extend(new_movies)
        
        for movie in new_movies:
            text = self._create_movie_text(movie)
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
        
        # Rebuild index
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

# Global BM25 retriever instance
bm25_retriever = BM25Retriever()
