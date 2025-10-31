"""LangGraph workflow implementation for movie recommendations."""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_tools.state import GraphState
from langgraph_tools.tools import (
    query_intent_classifier,
    temporal_query_parser_tool,
    intelligent_search_tmdb,
    semantic_wiki_retrieval,
    confidence_scorer
)
from retrieval.hybrid_retriever import hybrid_retriever
from retrieval.reranker import reranker
from retrieval.diversity_filter import diversity_filter
from data_sources.wikipedia_client import wiki_client
from config import config

class MovieRecommendationGraph:
    """LangGraph-based movie recommendation system."""
    
    def __init__(self):
        """Initialize the recommendation graph."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # Use Gemini 2.0 Flash
            temperature=0,
            google_api_key=config.GOOGLE_API_KEY
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("parse_temporal", self.parse_temporal_node)
        workflow.add_node("retrieve_parallel", self.retrieve_parallel_node)
        workflow.add_node("aggregate_rerank", self.aggregate_rerank_node)
        workflow.add_node("generate_explanations", self.generate_explanations_node)
        
        # Define edges
        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "parse_temporal")
        workflow.add_edge("parse_temporal", "retrieve_parallel")
        workflow.add_edge("retrieve_parallel", "aggregate_rerank")
        workflow.add_edge("aggregate_rerank", "generate_explanations")
        workflow.add_edge("generate_explanations", END)
        
        return workflow.compile()
    
    def classify_intent_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Classify query intent with LLM."""
        query = state["query"]
        
        try:
            # Use invoke for LangChain tools
            result = query_intent_classifier.invoke({"query": query})
            
            # Store the full result as intent (it's already a dict)
            return {
                "intent": result,  # Store full analysis dict
                "intent_confidence": result.get("confidence", 0.5),
                "extracted_genres": result.get("genres", []),
                "errors": []
            }
        except Exception as e:
            print(f"  ⚠ Intent classification error: {e}")
            return {
                "intent": {
                    "intent": "exploration",
                    "confidence": 0.5,
                    "genres": [],
                    "mood": "unknown",
                    "themes": [],
                    "keywords": []
                },
                "intent_confidence": 0.5,
                "extracted_genres": [],
                "errors": [f"Intent classification error: {str(e)}"]
            }
    
    def parse_temporal_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Parse temporal constraints."""
        query = state["query"]
        
        try:
            # Use invoke for LangChain tools
            result = temporal_query_parser_tool.invoke({"query": query})
            
            return {
                "temporal_constraints": result,
                "errors": []
            }
        except Exception as e:
            return {
                "temporal_constraints": {"start_year": None, "end_year": None},
                "errors": [f"Temporal parsing error: {str(e)}"]
            }
    
    def retrieve_parallel_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Parallel retrieval using LLM-analyzed intent."""
        query = state["query"]
        intent_data = state.get("intent", {})
        temporal = state.get("temporal_constraints", {})
        
        # Extract LLM analysis results
        genres = intent_data.get("genres", [])
        themes = intent_data.get("themes", [])
        mood = intent_data.get("mood")
        keywords = intent_data.get("keywords", [])
        
        errors = []
        tmdb_results = []
        wiki_data = {}
        hybrid_results = []
        
        # 1. Hybrid retrieval (FAISS + BM25) - RAG component
        print("  → Hybrid retrieval (Vector + BM25)...")
        try:
            hybrid_results = hybrid_retriever.search(
                query=query,
                k=50
            )
            print(f"    ✓ Found {len(hybrid_results)} movies from local database")
        except Exception as e:
            print(f"    ⚠ Hybrid retrieval failed: {str(e)}")
            errors.append(f"Hybrid retrieval error: {str(e)}")
        
        # 2. Intelligent TMDb search using LLM analysis
        print("  → Intelligent TMDb API search...")
        try:
            # Use invoke for LangChain tools
            tmdb_results = intelligent_search_tmdb.invoke({
                "query": query,
                "genres": genres,
                "themes": themes,
                "mood": mood,
                "keywords": keywords,
                "temporal_constraints": temporal if (temporal and (temporal.get("start_year") or temporal.get("end_year"))) else None,
                "max_results": 30
            })
            print(f"    ✓ Found {len(tmdb_results)} movies from TMDb")
        except Exception as e:
            print(f"    ⚠ TMDb API failed: {str(e)}")
            errors.append(f"TMDb search error: {str(e)}")
        
        # 3. Wikipedia enrichment for ALL results (for better plot matching)
        print("  → Wikipedia enrichment (fetching plots for matching)...")
        try:
            enriched_count = 0
            for movie in tmdb_results:  # Enrich all, not just top 5
                title = movie.get("title")
                year = movie.get("release_date", "")[:4] if movie.get("release_date") else None
                
                if title:
                    wiki_info = wiki_client.get_movie_info(title, int(year) if year else None)
                    if wiki_info:
                        wiki_data[movie.get("id")] = wiki_info
                        # Add to movie data for re-ranking
                        movie["wiki_data"] = wiki_info
                        # Add plot as searchable text for better matching
                        if wiki_info.get("plot"):
                            movie["wiki_plot"] = wiki_info["plot"]
                        if wiki_info.get("themes"):
                            movie["wiki_themes"] = wiki_info["themes"]
                        enriched_count += 1
            print(f"  ✓ Wikipedia enriched {enriched_count} movies with plots and themes")
        except Exception as e:
            print(f"  ⚠ Wikipedia enrichment failed: {str(e)}")
            errors.append(f"Wikipedia enrichment error: {str(e)}")
        
        return {
            "tmdb_results": tmdb_results,
            "hybrid_results": hybrid_results,
            "wiki_data": wiki_data,
            "errors": errors
        }
    
    def aggregate_rerank_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Aggregate hybrid + TMDb results, re-rank, and apply diversity filter."""
        query = state["query"]
        tmdb_results = state.get("tmdb_results", [])
        hybrid_results = state.get("hybrid_results", [])
        genres = state.get("extracted_genres", [])
        
        errors = []
        
        print("  → Aggregating and re-ranking results...")
        try:
            # Merge hybrid (RAG) and TMDb (API) results
            all_results = {}
            
            # Add hybrid results (from vector store + BM25)
            for movie, score in hybrid_results:
                movie_id = movie.get("id")
                all_results[movie_id] = (movie, score)
            
            # Add TMDb results (with lower initial score if duplicate)
            for movie in tmdb_results:
                movie_id = movie.get("id")
                if movie_id not in all_results:
                    all_results[movie_id] = (movie, 0.7)
            
            if not all_results:
                return {
                    "reranked_results": [],
                    "diverse_results": [],
                    "confidence_scores": {},
                    "errors": ["No movies found matching your query. Try different search terms."]
                }
            
            # Convert to candidates list
            candidates = list(all_results.values())
            print(f"  ✓ Combined {len(candidates)} unique movies")
            
            # Re-rank
            reranked = reranker.rerank(
                candidates,
                query=query,
                query_genres=genres,
                max_results=20
            )
            print(f"  ✓ Re-ranked to top {len(reranked)} movies")
            
            # Apply diversity filter
            diverse = diversity_filter.apply_mmr(reranked, query=query, k=10)
            print(f"  ✓ Diversified to {len(diverse)} movies")
            
            # Calculate confidence scores
            confidence_scores = {}
            for movie, score in diverse:
                movie_id = movie.get("id")
                conf_result = confidence_scorer.invoke({"movie_data": movie})
                confidence_scores[movie_id] = conf_result.get("confidence_score", 0.5)
            
            return {
                "reranked_results": reranked,
                "diverse_results": diverse,
                "confidence_scores": confidence_scores,
                "errors": errors
            }
        
        except Exception as e:
            return {
                "reranked_results": [],
                "diverse_results": [],
                "confidence_scores": {},
                "errors": [f"Aggregation error: {str(e)}"]
            }
    
    def generate_explanations_node(self, state: GraphState) -> Dict[str, Any]:
        """Node: Generate explanations and final recommendations."""
        diverse_results = state.get("diverse_results", [])
        confidence_scores = state.get("confidence_scores", {})
        query = state["query"]
        
        recommendations = []
        explanations = []
        
        try:
            for movie, score in diverse_results:
                movie_id = movie.get("id")
                confidence = confidence_scores.get(movie_id, 0.5)
                
                # Build recommendation
                rec = {
                    "id": movie_id,
                    "title": movie.get("title"),
                    "year": movie.get("release_date", "")[:4] if movie.get("release_date") else "N/A",
                    "rating": movie.get("vote_average", 0),
                    "genres": [
                        g.get("name", g) if isinstance(g, dict) else str(g)
                        for g in movie.get("genres", [])
                    ],
                    "overview": movie.get("overview", ""),
                    "poster_path": movie.get("poster_path"),
                    "match_score": score,
                    "confidence": confidence
                }
                
                # Generate explanation
                explanation = self._generate_explanation(movie, query, score, confidence)
                
                recommendations.append(rec)
                explanations.append(explanation)
            
            return {
                "recommendations": recommendations,
                "explanations": explanations,
                "errors": []
            }
        
        except Exception as e:
            return {
                "recommendations": [],
                "explanations": [],
                "errors": [f"Explanation generation error: {str(e)}"]
            }
    
    def _generate_explanation(
        self,
        movie: Dict,
        query: str,
        match_score: float,
        confidence: float
    ) -> str:
        """Generate explanation for a recommendation."""
        title = movie.get("title", "Unknown")
        year = movie.get("release_date", "")[:4] if movie.get("release_date") else "N/A"
        rating = movie.get("vote_average", 0)
        genres = movie.get("genres", [])
        genre_names = [g.get("name", g) if isinstance(g, dict) else str(g) for g in genres]
        
        explanation = f"**{title} ({year})** - Match: {match_score:.1%}\n"
        explanation += f"Rating: {rating}/10 ⭐ | Confidence: {confidence:.1%}\n"
        explanation += f"Genres: {', '.join(genre_names)}\n\n"
        
        # Add Wikipedia context if available
        wiki_data = movie.get("wiki_data", {})
        if wiki_data.get("themes"):
            explanation += f"Themes: {wiki_data['themes'][:150]}...\n\n"
        
        overview = movie.get("overview", "")
        if overview:
            explanation += f"{overview[:250]}...\n"
        
        return explanation
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the recommendation workflow."""
        initial_state = {
            "query": query,
            "errors": []
        }
        
        result = self.graph.invoke(initial_state)
        return result

def create_recommendation_graph():
    """Create and return a compiled recommendation graph."""
    return MovieRecommendationGraph().graph

# Global graph instance (deprecated, use create_recommendation_graph instead)
recommendation_graph = MovieRecommendationGraph()
