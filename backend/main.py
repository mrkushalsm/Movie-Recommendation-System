import os
import sys
import logging
from dotenv import load_dotenv
from langsmith import Client
from typing import List, Dict, Any
from langgraph_tools.graph import create_recommendation_graph
from utils.cache_manager import cache_manager
from data_sources.tmdb_client import tmdb_client
from vector_store.faiss_store import FAISSVectorStore as FAISSStore
from retrieval.bm25_retriever import BM25Retriever


# ======================================================
# LangSmith + Logging Setup
# ======================================================

# Load environment variables from .env
load_dotenv()

# Configure LangSmith client if tracing enabled
if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    try:
        client = Client()  # Automatically uses LANGSMITH_API_KEY
        logging.info(
            f"✅ LangSmith tracing enabled for project: "
            f"{os.getenv('LANGSMITH_PROJECT', 'default')}"
        )
    except Exception as e:
        logging.warning(f"⚠ Failed to initialize LangSmith client: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================
# Main Movie Recommendation Application
# ======================================================

class MovieRecommendationApp:
    """Main application class."""

    def __init__(self):
        """Initialize the application."""
        print("Movie Recommendation System initialized")
        print("Loading vector store and BM25 index...")
        self.load_indexes()
        print("Using RAG (Vector Store + BM25) with TMDb API fallback")
        self.graph = create_recommendation_graph()

    def load_indexes(self):
        """Load or initialize FAISS vector store and BM25 index."""
        try:
            # Try to load existing indexes
            self.vector_store = FAISSStore()
            self.vector_store.load()
            count = self.vector_store.index.ntotal if self.vector_store.index else 0
            print(f"  ✓ Loaded {count} movies from vector store")

            self.bm25_retriever = BM25Retriever()
            self.bm25_retriever.load()
            print(f"  ✓ Loaded BM25 index")
        except (FileNotFoundError, Exception):
            print("  ℹ No existing vector store found. Creating new one...")
            print("  ℹ Database will grow dynamically from your queries")
            # Create new empty indexes
            self.vector_store = FAISSStore()
            self.bm25_retriever = BM25Retriever()

    def get_recommendations(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Get movie recommendations using LangGraph workflow."""
        initial_state = {
            "query": query,
            "intent": {},
            "temporal_constraints": {},
            "tmdb_results": [],
            "final_recommendations": [],
            "explanations": [],
            "warnings": [],
            "top_k": top_k
        }

        # Run the LangGraph workflow
        final_state = self.graph.invoke(initial_state)

        # Save new movies dynamically to vector store
        self._save_new_movies_to_store(final_state)

        return final_state

    def _save_new_movies_to_store(self, state: Dict[str, Any]):
        """Save newly discovered movies to vector store and BM25 index."""
        try:
            tmdb_results = state.get("tmdb_results", [])
            if not tmdb_results or not self.vector_store:
                return

            new_movies = []
            for movie in tmdb_results:
                movie_id = movie.get("id")
                if movie_id and not self._movie_exists(movie_id):
                    new_movies.append(movie)
                    self.vector_store.add_movie(movie)

            if new_movies:
                self.bm25_retriever.add_movies(new_movies)
                self.vector_store.save()
                self.bm25_retriever.save()
                print(f"\n💾 Saved {len(new_movies)} new movies to database")

        except Exception as e:
            print(f"\n⚠ Error saving movies: {e}")

    def _movie_exists(self, movie_id: int) -> bool:
        """Check if a movie already exists in the vector store."""
        # Simple placeholder; replace with more sophisticated check if needed
        return False

    def run_query(self, query: str):
        """Process a single user query and display results."""
        print(f"\n🔍 Query: {query}\n")
        print("Processing...\n")

        result = self.get_recommendations(query)
        self.display_results(result)

    def display_results(self, result: Dict[str, Any]):
        """Display movie recommendation results."""
        print("=" * 70)
        print("MOVIE RECOMMENDATIONS")
        print("=" * 70)

        # Handle warnings or errors
        if result.get("errors"):
            print("\n⚠ Warnings:")
            for error in result["errors"]:
                print(f"  - {error}")

        # Display recommendations
        recommendations = result.get("recommendations", [])
        explanations = result.get("explanations", [])

        if not recommendations:
            print("\nNo recommendations found.")
            print("\nTips:")
            print("  - Try different or broader search terms")
            print("  - Check your internet connection")
            print("  - Verify TMDb API is working")
            return

        for i, (rec, explanation) in enumerate(zip(recommendations, explanations), 1):
            movie = rec.get("movie", rec)
            print(
                f"\n{i}. {movie.get('title', 'Unknown')} "
                f"({movie.get('release_date', 'N/A')[:4] if movie.get('release_date') else rec.get('year', 'N/A')})"
            )
            if explanation:
                print(f"   {explanation}")

    def interactive_mode(self):
        """Interactive console mode."""
        print("\n" + "=" * 70)
        print("MOVIE RECOMMENDATION SYSTEM - Interactive Mode")
        print("=" * 70)
        print("\nEnter your movie preferences and get personalized recommendations!")
        print("Type 'exit' or 'quit' to stop.\n")
        print("Example queries:")
        print("  1. Best sci-fi movies from 2020")
        print("  2. Thrillers like Inception")
        print("  3. Classic films from the 80s")
        print("  4. Recent action movies with high ratings")
        print("  5. Horror movies from the 90s\n")

        while True:
            try:
                query = input("Your query: ").strip()

                if query.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye! 🎬")
                    break

                if not query:
                    continue

                self.run_query(query)
                print("\n" + "-" * 70 + "\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 🎬")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue

    def batch_mode(self, queries: List[str]):
        """Batch mode for predefined queries."""
        print("\n" + "=" * 70)
        print("MOVIE RECOMMENDATION SYSTEM - Batch Mode")
        print("=" * 70)

        for i, query in enumerate(queries, 1):
            print(f"\n\n{'#' * 70}")
            print(f"Query {i}/{len(queries)}")
            print(f"{'#' * 70}")
            self.run_query(query)


# ======================================================
# Entry Point
# ======================================================

def main():
    """Main entry point."""
    app = MovieRecommendationApp()

    # Check for command-line arguments
    if len(sys.argv) > 1:
        # Batch mode with query from command line
        query = " ".join(sys.argv[1:])
        app.run_query(query)
    else:
        # Interactive mode
        app.interactive_mode()


if __name__ == "__main__":
    main()
