"""FAISS-based vector store for movie embeddings."""
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from config import config
from vector_store.embeddings import embedding_manager

class FAISSVectorStore:
    """FAISS vector store for efficient similarity search."""
    
    def __init__(self, dimension: int = None):
        """Initialize FAISS index."""
        self.dimension = dimension or embedding_manager.dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine after normalization)
        self.movie_metadata: List[Dict] = []
        self.id_to_index: Dict[int, int] = {}
        self.store_path = config.VECTOR_STORE_DIR / "faiss_index.bin"
        self.metadata_path = config.VECTOR_STORE_DIR / "movie_metadata.pkl"
    
    def add_movies(self, movies: List[Dict], regenerate_embeddings: bool = False):
        """
        Add movies to the vector store.
        
        Args:
            movies: List of movie dictionaries
            regenerate_embeddings: Force regeneration of embeddings
        """
        embeddings_list = []
        
        for movie in movies:
            movie_id = movie.get("id")
            
            # Skip if already indexed
            if not regenerate_embeddings and movie_id in self.id_to_index:
                continue
            
            # Generate or retrieve embedding
            if "embedding" in movie and not regenerate_embeddings:
                embedding = np.array(movie["embedding"])
            else:
                embedding = embedding_manager.generate_movie_embedding(movie)
                movie["embedding"] = embedding.tolist()
            
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            
            embeddings_list.append(embedding)
            
            # Store metadata
            current_index = len(self.movie_metadata)
            self.movie_metadata.append(movie)
            self.id_to_index[movie_id] = current_index
        
        # Add to FAISS index
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype('float32')
            self.index.add(embeddings_array)
            print(f"Added {len(embeddings_list)} movies to vector store")
    
    def search(self, query: str, k: int = 20) -> List[Tuple[Dict, float]]:
        """
        Search for similar movies using semantic search.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (movie_dict, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = embedding_manager.generate_query_embedding(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.movie_metadata):
                movie = self.movie_metadata[idx]
                similarity = float(distance)  # Already cosine similarity due to normalization
                results.append((movie, similarity))
        
        return results
    
    def search_by_movie_id(self, movie_id: int, k: int = 10) -> List[Tuple[Dict, float]]:
        """Find movies similar to a given movie."""
        if movie_id not in self.id_to_index:
            return []
        
        movie_index = self.id_to_index[movie_id]
        movie = self.movie_metadata[movie_index]
        
        # Use movie's embedding
        if "embedding" in movie:
            embedding = np.array(movie["embedding"])
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.reshape(1, -1).astype('float32')
            
            k = min(k + 1, self.index.ntotal)  # +1 to exclude the movie itself
            distances, indices = self.index.search(embedding, k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.movie_metadata) and idx != movie_index:
                    similar_movie = self.movie_metadata[idx]
                    similarity = float(distance)
                    results.append((similar_movie, similarity))
            
            return results[:k-1]
        
        return []
    
    def save(self):
        """Save index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, str(self.store_path))
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.movie_metadata,
                'id_to_index': self.id_to_index
            }, f)
        
        print(f"Saved vector store with {self.index.ntotal} movies")
    
    def load(self) -> bool:
        """Load index and metadata from disk."""
        if not self.store_path.exists() or not self.metadata_path.exists():
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.store_path))
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.movie_metadata = data['metadata']
                self.id_to_index = data['id_to_index']
            
            print(f"Loaded vector store with {self.index.ntotal} movies")
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def clear(self):
        """Clear the vector store."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.movie_metadata = []
        self.id_to_index = {}

# Global vector store instance
vector_store = FAISSVectorStore()
