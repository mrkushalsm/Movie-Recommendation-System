"""Initialize vector store and BM25 index from TMDb data."""
import os
from tqdm import tqdm
from data_sources.tmdb_client import tmdb_client
from vector_store.faiss_store import FAISSVectorStore
from retrieval.bm25_retriever import BM25Retriever
from config import config

def fetch_popular_movies(max_pages: int = 50) -> list:
    """Fetch popular movies from TMDb."""
    print(f"Fetching popular movies from TMDb (up to {max_pages} pages)...")
    
    all_movies = []
    for page in tqdm(range(1, max_pages + 1), desc="Fetching pages"):
        try:
            result = tmdb_client.discover_movies(
                sort_by="popularity.desc",
                page=page,
                vote_count_gte=100,  # Only movies with sufficient votes
                vote_average_gte=5.0  # Only decent movies
            )
            
            movies = result.get("results", [])
            if not movies:
                break
                
            all_movies.extend(movies)
            
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
    
    print(f"Fetched {len(all_movies)} movies")
    return all_movies

def enrich_movies(movies: list) -> list:
    """Enrich movies with detailed metadata."""
    print("\nEnriching movies with detailed metadata...")
    
    enriched = []
    for movie in tqdm(movies, desc="Enriching"):
        try:
            movie_id = movie.get("id")
            if not movie_id:
                continue
            
            # Get detailed info
            details = tmdb_client.get_movie_details(movie_id)
            if details:
                movie.update(details)
            
            # Get credits
            credits = tmdb_client.get_movie_credits(movie_id)
            if credits:
                movie["cast"] = credits.get("cast", [])[:10]  # Top 10 cast
                movie["crew"] = credits.get("crew", [])
            
            enriched.append(movie)
            
        except Exception as e:
            print(f"Error enriching movie {movie.get('title')}: {e}")
            continue
    
    print(f"Enriched {len(enriched)} movies")
    return enriched

def initialize_vector_store(movies: list):
    """Initialize FAISS vector store."""
    print("\nInitializing FAISS vector store...")
    
    vector_store = FAISSVectorStore()
    
    # Add all movies at once
    try:
        vector_store.add_movies(movies)
    except Exception as e:
        print(f"Error adding movies: {e}")
        return
    
    # Save
    vector_store.save()
    print(f"✓ Vector store saved with {vector_store.index.ntotal} movies")

def initialize_bm25(movies: list):
    """Initialize BM25 index."""
    print("\nInitializing BM25 index...")
    
    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(movies)
    bm25_retriever.save()
    
    print(f"✓ BM25 index saved with {len(movies)} movies")

def main():
    """Main initialization process."""
    print("=" * 70)
    print("MOVIE RECOMMENDATION SYSTEM - Data Initialization")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Fetch popular movies from TMDb API")
    print("  2. Enrich with detailed metadata (cast, crew, etc.)")
    print("  3. Build FAISS vector store")
    print("  4. Build BM25 search index")
    print("\nThis may take 10-30 minutes depending on network speed.")
    print("=" * 70)
    
    # Fetch movies
    movies = fetch_popular_movies(max_pages=50)  # ~1000 movies
    
    if not movies:
        print("\n❌ Failed to fetch movies. Check your internet and TMDb API key.")
        return
    
    # Enrich
    enriched_movies = enrich_movies(movies)
    
    if not enriched_movies:
        print("\n❌ Failed to enrich movies.")
        return
    
    # Initialize stores
    initialize_vector_store(enriched_movies)
    initialize_bm25(enriched_movies)
    
    print("\n" + "=" * 70)
    print("✓ INITIALIZATION COMPLETE!")
    print("=" * 70)
    print("\nYou can now run: python main.py")

if __name__ == "__main__":
    main()
