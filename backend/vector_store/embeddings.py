"""Embedding generation and vector store management."""
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from config import config
import pickle
from pathlib import Path

class EmbeddingManager:
    """Manage embeddings for movies using sentence transformers."""
    
    def __init__(self):
        """Initialize embedding model."""
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def generate_movie_embedding(self, movie_data: Dict) -> np.ndarray:
        """
        Generate embedding for a movie from its metadata.
        
        Combines: title, overview, genres, keywords, themes
        """
        text_components = []
        
        # Title
        if movie_data.get("title"):
            text_components.append(f"Title: {movie_data['title']}")
        
        # Overview
        if movie_data.get("overview"):
            text_components.append(f"Plot: {movie_data['overview']}")
        
        # Genres
        genres = movie_data.get("genres", [])
        if genres:
            genre_names = [g.get("name", g) if isinstance(g, dict) else str(g) for g in genres]
            text_components.append(f"Genres: {', '.join(genre_names)}")
        
        # Keywords
        keywords = movie_data.get("keywords", [])
        if keywords:
            keyword_names = [k.get("name", k) if isinstance(k, dict) else str(k) for k in keywords[:10]]
            text_components.append(f"Keywords: {', '.join(keyword_names)}")
        
        # Wikipedia themes
        if movie_data.get("wiki_themes"):
            text_components.append(f"Themes: {movie_data['wiki_themes'][:500]}")
        
        # Combine all components
        combined_text = " | ".join(text_components)
        
        # Generate embedding
        return self.model.encode(combined_text, convert_to_numpy=True)
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query."""
        return self.model.encode(query, convert_to_numpy=True)
    
    def batch_generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    
    def batch_cosine_similarity(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and multiple embeddings."""
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute similarities
        return np.dot(embeddings_norm, query_norm)

# Global embedding manager
embedding_manager = EmbeddingManager()
