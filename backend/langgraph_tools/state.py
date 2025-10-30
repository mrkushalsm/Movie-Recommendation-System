"""LangGraph workflow state definition."""
from typing import TypedDict, List, Dict, Optional, Any, Annotated
import operator

class GraphState(TypedDict):
    """State for the recommendation graph."""
    
    # Input
    query: str
    
    # Intent classification
    intent: Optional[str]
    intent_confidence: Optional[float]
    extracted_genres: Optional[List[str]]
    
    # Temporal parsing
    temporal_constraints: Optional[Dict[str, int]]
    
    # Retrieval
    tmdb_results: Optional[List[Dict]]
    hybrid_results: Optional[List[tuple]]  # From FAISS + BM25 (RAG)
    wiki_data: Optional[Dict[str, Dict]]
    
    # Re-ranking
    reranked_results: Optional[List[tuple]]
    diverse_results: Optional[List[tuple]]
    
    # Final output
    recommendations: Optional[List[Dict]]
    explanations: Optional[List[str]]
    
    # Metadata
    confidence_scores: Optional[Dict[int, float]]
    errors: Annotated[List[str], operator.add]
