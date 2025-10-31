"""Chroma-based vector store for movie embeddings."""
import chromadb
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from config import config
from vector_store.embeddings import embedding_manager

class ChromaVectorStore:
    """Chroma vector store for efficient similarity search."""
    
    def __init__(self, collection_name: str = "movies"):
        """Initialize Chroma vector store."""
        self.collection_name = collection_name
        self.chroma_dir = config.VECTOR_STORE_DIR / "chroma_db"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client with persistent storage
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.movie_metadata: Dict[str, Dict] = {}  # Store full movie data
    
    def add_movies(self, movies: List[Dict], regenerate_embeddings: bool = False):
        """
        Add movies to the vector store.
        
        Args:
            movies: List of movie dictionaries
            regenerate_embeddings: Force regeneration of embeddings
        """
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for movie in movies:
            movie_id = str(movie.get("id"))
            
            # Skip if already indexed (unless regenerating)
            try:
                existing = self.collection.get(ids=[movie_id])
                if existing['ids'] and not regenerate_embeddings:
                    continue
            except:
                pass
            
            # Generate or retrieve embedding
            if "embedding" in movie and not regenerate_embeddings:
                embedding = movie["embedding"]
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
            else:
                embedding = embedding_manager.generate_movie_embedding(movie)
                if isinstance(embedding, np.ndarray):
                    movie["embedding"] = embedding.tolist()
            
            # Normalize for cosine similarity
            if isinstance(embedding, np.ndarray):
                embedding = embedding / np.linalg.norm(embedding)
                embedding = embedding.tolist()
            else:
                embedding_np = np.array(embedding)
                embedding_np = embedding_np / np.linalg.norm(embedding_np)
                embedding = embedding_np.tolist()
            
            # Prepare metadata (exclude embedding to save space)
            metadata = {k: v for k, v in movie.items() if k != "embedding"}
            
            # Convert non-serializable types
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    try:
                        # Try to keep as-is (Chroma can handle JSON)
                        pass
                    except:
                        metadata[key] = str(value)
                elif not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
            
            # Prepare document text for search
            title = movie.get("title", "Unknown")
            overview = movie.get("overview", "")
            genres = movie.get("genres", [])
            if isinstance(genres, list):
                genres_str = " ".join([g if isinstance(g, str) else g.get("name", "") for g in genres])
            else:
                genres_str = str(genres)
            
            document = f"{title} {overview} {genres_str}".strip()
            
            ids.append(movie_id)
            embeddings.append(embedding)
            metadatas.append(metadata)
            documents.append(document)
            
            # Store full movie data locally
            self.movie_metadata[movie_id] = movie
        
        # Add to Chroma collection
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            print(f"Added {len(ids)} movies to Chroma vector store")
    
    def search(self, query: str, k: int = 20) -> List[Tuple[Dict, float]]:
        """
        Search for similar movies using semantic search.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (movie_dict, similarity_score) tuples
        """
        if self.collection.count() == 0:
            return []
        
        # Generate query embedding
        query_embedding = embedding_manager.generate_query_embedding(query)
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
        query_embedding_list = query_embedding_norm.tolist()
        
        # Search using Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=k,
            include=["embeddings", "metadatas", "distances"]
        )
        
        # Convert results
        output = []
        if results['ids'] and len(results['ids']) > 0:
            for idx, (movie_id, metadata, distance) in enumerate(
                zip(results['ids'][0], results['metadatas'][0], results['distances'][0])
            ):
                # Distance in Chroma cosine is 1 - similarity, convert back
                similarity = 1 - distance
                
                # Reconstruct movie dict from metadata
                if movie_id in self.movie_metadata:
                    movie = self.movie_metadata[movie_id]
                else:
                    movie = metadata
                
                output.append((movie, similarity))
        
        return output
    
    def search_by_movie_id(self, movie_id: int, k: int = 10) -> List[Tuple[Dict, float]]:
        """Find movies similar to a given movie."""
        movie_id_str = str(movie_id)
        
        if movie_id_str not in self.movie_metadata:
            return []
        
        movie = self.movie_metadata[movie_id_str]
        
        # Use movie's embedding
        if "embedding" in movie:
            embedding = movie["embedding"]
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            embedding = embedding / np.linalg.norm(embedding)
            embedding_list = embedding.tolist()
            
            # Search similar movies (excluding itself)
            results = self.collection.query(
                query_embeddings=[embedding_list],
                n_results=k + 1,  # +1 to account for the movie itself
                include=["metadatas", "distances"]
            )
            
            output = []
            if results['ids'] and len(results['ids']) > 0:
                for movie_id_result, metadata, distance in zip(
                    results['ids'][0], results['metadatas'][0], results['distances'][0]
                ):
                    # Skip the movie itself
                    if movie_id_result == movie_id_str:
                        continue
                    
                    similarity = 1 - distance
                    
                    if movie_id_result in self.movie_metadata:
                        result_movie = self.movie_metadata[movie_id_result]
                    else:
                        result_movie = metadata
                    
                    output.append((result_movie, similarity))
            
            return output[:k]
        
        return []
    
    def save(self):
        """Save vector store (Chroma handles persistence automatically)."""
        print(f"Saved Chroma vector store with {self.collection.count()} movies")
    
    def load(self) -> bool:
        """Load vector store (Chroma handles persistence automatically)."""
        try:
            count = self.collection.count()
            if count > 0:
                print(f"Loaded Chroma vector store with {count} movies")
                return True
            return True  # Always return True since collection exists
        except Exception as e:
            print(f"Error loading Chroma vector store: {e}")
            return False
    
    def clear(self):
        """Clear the vector store."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.movie_metadata = {}
            print("Cleared Chroma vector store")
        except Exception as e:
            print(f"Error clearing Chroma vector store: {e}")

# Global vector store instance
vector_store = ChromaVectorStore()
