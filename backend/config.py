"""Configuration management for the movie recommendation system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class."""
    
        # API Keys
    TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
    TMDB_ACCESS_TOKEN: str = os.getenv("TMDB_ACCESS_TOKEN", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")
    
    # Cache Configuration
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    
    # Data paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    VECTOR_STORE_DIR = DATA_DIR / "vector_store"
    CACHE_DIR = DATA_DIR / "cache"
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Recommendation parameters
    MAX_RECOMMENDATIONS = 10
    MIN_RATING_COUNT = 50
    MIN_VOTE_AVERAGE = 6.0
    
    # Re-ranking weights
    SEMANTIC_WEIGHT = 0.3
    GENRE_WEIGHT = 0.2
    RATING_WEIGHT = 0.2
    RECENCY_WEIGHT = 0.15
    POPULARITY_WEIGHT = 0.15

config = Config()
