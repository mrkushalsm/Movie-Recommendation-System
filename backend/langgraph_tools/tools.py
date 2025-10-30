"""LangGraph tool functions for the movie recommendation system."""
from typing import Dict, List, Any, Optional, Annotated
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from data_sources.tmdb_client import tmdb_client
from data_sources.wikipedia_client import wiki_client
from data_sources.ratings_client import ratings_client
from retrieval.hybrid_retriever import hybrid_retriever
from retrieval.reranker import reranker
from retrieval.diversity_filter import diversity_filter
from utils.temporal_parser import temporal_parser
from vector_store.faiss_store import vector_store
from config import config
import json

# Initialize LLM for intelligent analysis
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # Use Gemini 2.0 Flash
    temperature=0,
    google_api_key=config.GOOGLE_API_KEY
)

@tool
def query_intent_classifier(query: str) -> Dict[str, Any]:
    """
    Use LLM to deeply analyze user query for intent, mood, themes, and preferences.
    
    This is the INTELLIGENCE layer - extracts:
    - Intent (exploration, similar_to, mood_based, etc.)
    - Mood/Tone (dark, uplifting, suspenseful, etc.)
    - Themes (war, family, revenge, identity, etc.)
    - Genres (action, drama, thriller, etc.)
    - Specific requirements (actors, directors, era, etc.)
    
    Args:
        query: User's natural language query
    
    Returns:
        Structured analysis with all extracted information
    """
    
    prompt = f"""Analyze this movie recommendation query and extract ALL relevant information:

Query: "{query}"

Extract and return a JSON object with:
{{
  "intent": "exploration|similar_to|mood_based|comparison|direct_match",
  "confidence": 0.0-1.0,
  "genres": ["genre1", "genre2"],
  "mood": "overall emotional tone (e.g., dark, uplifting, tense, nostalgic)",
  "themes": ["theme1", "theme2"] (e.g., war, revenge, family, identity),
  "keywords": ["keyword1", "keyword2"] (important concepts),
  "mentioned_movies": ["movie1"] (if any specific movies mentioned),
  "mentioned_people": ["person1"] (actors/directors if mentioned),
  "era_preference": "decade or time period if mentioned",
  "rating_preference": "high|medium|any",
  "specific_requirements": "any other specific needs"
}}

Examples:
- "war movies about indian soldiers" → mood: "intense, patriotic", themes: ["war", "military", "patriotism"], keywords: ["soldiers", "indian", "combat"]
- "something dark like Se7en" → mood: "dark, disturbing", themes: ["crime", "psychology"], mentioned_movies: ["Se7en"]
- "uplifting family films from the 80s" → mood: "uplifting, heartwarming", themes: ["family"], era_preference: "1980s"

Return ONLY the JSON, no other text."""

    try:
        print(f"  🧠 Analyzing query with Gemini...")
        response = llm.invoke(prompt)
        
        # Parse JSON from response
        content = response.content
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        analysis = json.loads(content)
        print(f"  ✓ Intent: {analysis.get('intent')}, Mood: {analysis.get('mood')}")
        print(f"  ✓ Themes: {analysis.get('themes')}")
        
        return analysis
        
    except Exception as e:
        print(f"  ⚠ LLM analysis failed: {e}, using fallback")
        # Fallback to simple keyword extraction
        query_lower = query.lower()
        
        genres = []
        genre_keywords = {
            "action", "comedy", "drama", "thriller", "horror", "sci-fi",
            "romance", "adventure", "fantasy", "mystery", "crime", "war"
        }
        
        for genre in genre_keywords:
            if genre in query_lower:
                genres.append(genre.title())
        
        return {
            "intent": "exploration",
            "confidence": 0.5,
            "genres": genres,
            "mood": "unknown",
            "themes": [],
            "keywords": query_lower.split(),
            "original_query": query
        }


@tool
def temporal_query_parser_tool(query: str) -> Dict[str, Optional[int]]:
    """
    Extract temporal constraints from query.
    
    Examples:
    - "movies from 2020" -> {start_year: 2020, end_year: 2020}
    - "90s action films" -> {start_year: 1990, end_year: 1999}
    - "recent thrillers" -> {start_year: 2023, end_year: 2025}
    
    Args:
        query: User's query string
    
    Returns:
        Dict with start_year and end_year (None if not found)
    """
    return temporal_parser.parse(query)


@tool
def intelligent_search_tmdb(
    query: str,
    genres: Optional[List[str]] = None,
    themes: Optional[List[str]] = None,
    mood: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    temporal_constraints: Optional[Dict[str, Optional[int]]] = None,
    min_rating: float = 5.5,
    max_results: int = 30
) -> List[Dict]:
    """
    Intelligent TMDb search using LLM-extracted themes, mood, and keywords.
    
    This uses a multi-strategy approach:
    1. /discover with genre + temporal filters
    2. /search with enriched query (themes + mood + keywords)
    3. /discover with keyword IDs from TMDb
    
    Args:
        query: Original search query
        genres: List of genre names (from LLM)
        themes: List of thematic keywords (from LLM) - e.g., ["war", "patriotism"]
        mood: Emotional tone (from LLM) - e.g., "dark", "uplifting"
        keywords: Important concepts (from LLM)
        temporal_constraints: Dict with start_year/end_year
        min_rating: Minimum vote average
        max_results: Maximum results to return
    
    Returns:
        List of enriched movie dictionaries
    """
    all_movies = {}
    
    print(f"🔍 Intelligent TMDb Search:")
    print(f"  Query: '{query}'")
    print(f"  Genres: {genres}")
    print(f"  Themes: {themes}")
    print(f"  Mood: {mood}")
    print(f"  Keywords: {keywords}")
    
    # Strategy 1: /discover with genre + temporal filters
    if genres or temporal_constraints:
        print(f"  → Strategy 1: /discover with filters")
        filters = {
            "sort_by": "popularity.desc",
            "vote_average.gte": min_rating,
        }
        
        # Add temporal constraints
        if temporal_constraints:
            tmdb_dates = temporal_parser.format_for_tmdb(temporal_constraints)
            filters.update(tmdb_dates)
        
        # Add genre filters
        if genres:
            genre_map = {g["name"]: g["id"] for g in tmdb_client.get_genres().get("genres", [])}
            genre_ids = [str(genre_map.get(g)) for g in genres if g in genre_map]
            if genre_ids:
                filters["with_genres"] = ",".join(genre_ids)
        
        # Discover movies
        discovered_movies = tmdb_client.batch_discover(filters, max_pages=2, quality_filter=False)
        print(f"    ✓ Found {len(discovered_movies)} movies")
        for movie in discovered_movies:
            all_movies[movie.get("id")] = movie
    
    # Strategy 2: /search with enriched query (themes + mood)
    enriched_query_parts = [query]
    if themes:
        enriched_query_parts.extend(themes[:3])  # Top 3 themes
    if mood and mood != "unknown":
        enriched_query_parts.append(mood)
    
    enriched_query = " ".join(enriched_query_parts)
    
    print(f"  → Strategy 2: /search with enriched query: '{enriched_query}'")
    search_results = tmdb_client.search_movies(enriched_query)
    results = search_results.get("results", [])
    print(f"    ✓ Found {len(results)} movies")
    for movie in results[:25]:
        if movie.get("id") not in all_movies:
            all_movies[movie.get("id")] = movie
    
    # Strategy 3: Search with ORIGINAL query (most important!)
    # This ensures "indian soldiers" is actually searched
    if query not in enriched_query:
        print(f"  → Strategy 3: /search with original query: '{query}'")
        original_results = tmdb_client.search_movies(query)
        orig_movies = original_results.get("results", [])
        print(f"    ✓ Found {len(orig_movies)} movies")
        # Prioritize these results
        for movie in orig_movies[:30]:
            if movie.get("id") not in all_movies:
                all_movies[movie.get("id")] = movie
    
    # Strategy 4: If we have specific keywords, search for those
    if keywords and len(all_movies) < 20:
        print(f"  → Strategy 4: Searching individual keywords: {keywords[:3]}")
        for keyword in keywords[:3]:  # Top 3 keywords
            if keyword and len(keyword) > 3:  # Skip short keywords
                kw_results = tmdb_client.search_movies(keyword)
                for movie in kw_results.get("results", [])[:10]:
                    if movie.get("id") not in all_movies:
                        all_movies[movie.get("id")] = movie
    
    print(f"📊 Total unique movies found: {len(all_movies)}")
    
    # Convert to list and sort by popularity
    movies_list = sorted(
        all_movies.values(), 
        key=lambda x: x.get("popularity", 0), 
        reverse=True
    )
    
    # Apply minimal quality filter
    movies_list = [
        m for m in movies_list 
        if m.get("vote_count", 0) >= 10 and m.get("overview")
    ][:max_results]
    
    # Enrich top results
    enriched = []
    for movie in movies_list:
        try:
            enriched_movie = tmdb_client.enrich_movie_data(movie)
            enriched.append(enriched_movie)
        except Exception as e:
            # If enrichment fails, use basic data
            enriched.append(movie)
    
    return enriched


@tool
def semantic_wiki_retrieval(
    movie_title: str,
    year: Optional[int] = None,
    sections: List[str] = None
) -> Dict[str, str]:
    """
    Section-aware Wikipedia retrieval with context.
    
    Args:
        movie_title: Movie title
        year: Release year (optional)
        sections: Specific sections to retrieve (plot, themes, production)
    
    Returns:
        Dict with requested sections
    """
    sections = sections or ["plot", "themes", "production"]
    
    wiki_data = wiki_client.get_movie_info(movie_title, year)
    
    # Filter to requested sections
    filtered_data = {k: v for k, v in wiki_data.items() if k in sections}
    
    return filtered_data


@tool
def cross_validate_ratings(
    movie_title: str,
    year: Optional[int] = None,
    tmdb_rating: Optional[float] = None
) -> Dict[str, Any]:
    """
    Cross-validate ratings from multiple sources.
    
    Args:
        movie_title: Movie title
        year: Release year
        tmdb_rating: TMDb rating (0-10)
    
    Returns:
        Dict with ratings from different sources and consensus
    """
    return ratings_client.get_consensus_rating(movie_title, year, tmdb_rating)


@tool
def analyze_similarity_graph(movie_id: int, depth: int = 1) -> Dict[str, Any]:
    """
    Traverse movie relationship graph to build recommendation subgraph.
    
    Args:
        movie_id: TMDb movie ID
        depth: Graph traversal depth
    
    Returns:
        Dict with similar movies and relationship strengths
    """
    # Get similar movies from TMDb
    similar_tmdb = tmdb_client.get_similar_movies(movie_id).get("results", [])
    
    # Get similar movies from vector store
    similar_vector = vector_store.search_by_movie_id(movie_id, k=10)
    
    # Combine and deduplicate
    all_similar = {}
    
    for movie in similar_tmdb[:10]:
        movie_id = movie.get("id")
        all_similar[movie_id] = {
            "movie": movie,
            "similarity_score": 0.7,  # TMDb similarity (implicit)
            "source": "tmdb"
        }
    
    for movie, score in similar_vector:
        movie_id = movie.get("id")
        if movie_id in all_similar:
            # Average scores if from both sources
            all_similar[movie_id]["similarity_score"] = (
                all_similar[movie_id]["similarity_score"] + score
            ) / 2
            all_similar[movie_id]["source"] = "both"
        else:
            all_similar[movie_id] = {
                "movie": movie,
                "similarity_score": score,
                "source": "vector"
            }
    
    return {
        "source_movie_id": movie_id,
        "similar_movies": list(all_similar.values()),
        "count": len(all_similar)
    }


@tool
def diversity_filter_tool(
    candidates: str,  # JSON string of List[Tuple[Dict, float]]
    k: int = 10,
    lambda_param: float = 0.7
) -> str:
    """
    Apply MMR diversity filtering to prevent repetitive recommendations.
    
    Args:
        candidates: JSON string of (movie, score) tuples
        k: Number of diverse results
        lambda_param: Balance between relevance and diversity
    
    Returns:
        JSON string of diverse results
    """
    # Parse input
    candidates_list = json.loads(candidates)
    
    # Apply MMR
    diversity_filter.lambda_param = lambda_param
    diverse_results = diversity_filter.apply_mmr(
        [(c[0], c[1]) for c in candidates_list],
        k=k
    )
    
    return json.dumps([(movie, score) for movie, score in diverse_results])


@tool
def confidence_scorer(movie_data: Dict) -> Dict[str, Any]:
    """
    Assess data completeness and source agreement.
    
    Args:
        movie_data: Movie dictionary with metadata
    
    Returns:
        Dict with confidence score and quality metrics
    """
    completeness_score = 0.0
    max_score = 7.0
    
    # Check for essential fields
    if movie_data.get("title"):
        completeness_score += 1.0
    if movie_data.get("overview"):
        completeness_score += 1.0
    if movie_data.get("vote_count", 0) >= 50:
        completeness_score += 1.0
    if movie_data.get("vote_average", 0) > 0:
        completeness_score += 1.0
    if movie_data.get("genres"):
        completeness_score += 1.0
    if movie_data.get("poster_path"):
        completeness_score += 0.5
    if movie_data.get("release_date"):
        completeness_score += 0.5
    
    # Additional data bonus
    if movie_data.get("keywords"):
        completeness_score += 0.5
    if movie_data.get("credits"):
        completeness_score += 0.5
    
    confidence = completeness_score / max_score
    
    return {
        "confidence_score": confidence,
        "completeness_score": completeness_score,
        "max_score": max_score,
        "has_sufficient_data": confidence >= 0.6
    }


@tool
def explain_recommendation(
    movie: Dict,
    query: str,
    similarity_score: float,
    composite_score: float,
    evidence: Dict = None
) -> str:
    """
    Generate human-readable explanation for a recommendation.
    
    Args:
        movie: Movie dictionary
        query: Original query
        similarity_score: Semantic similarity score
        composite_score: Final composite score
        evidence: Additional evidence dict
    
    Returns:
        Human-readable explanation string
    """
    title = movie.get("title", "Unknown")
    year = movie.get("release_date", "")[:4] if movie.get("release_date") else "N/A"
    rating = movie.get("vote_average", 0)
    genres = movie.get("genres", [])
    genre_names = [g.get("name", g) if isinstance(g, dict) else str(g) for g in genres]
    
    explanation = f"**{title} ({year})**\n\n"
    explanation += f"**Rating:** {rating}/10 ⭐\n"
    explanation += f"**Genres:** {', '.join(genre_names)}\n\n"
    
    explanation += "**Why recommended:**\n"
    explanation += f"- Semantic similarity to your query: {similarity_score:.2%}\n"
    explanation += f"- Overall match score: {composite_score:.2%}\n"
    
    if evidence:
        if evidence.get("genre_match"):
            explanation += f"- Genre match: {evidence['genre_match']}\n"
        if evidence.get("rating_quality"):
            explanation += f"- High rating quality: {evidence['rating_quality']}\n"
        if evidence.get("popularity"):
            explanation += f"- Popularity: {evidence['popularity']}\n"
    
    overview = movie.get("overview", "")
    if overview:
        explanation += f"\n**Synopsis:** {overview[:200]}...\n"
    
    return explanation

# Export all tools
TOOLS = [
    query_intent_classifier,
    temporal_query_parser_tool,
    intelligent_search_tmdb,
    semantic_wiki_retrieval,
    cross_validate_ratings,
    analyze_similarity_graph,
    diversity_filter_tool,
    confidence_scorer,
    explain_recommendation
]
