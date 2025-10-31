# Quick Reference Guide - System Components

## 🔍 How the System Works (Simple Version)

```
User Query
    ↓
Extract Intent (LLM)
    ↓ [genres, mood, themes, keywords]
Search TMDb (4 strategies)
    ↓ [gets movie list]
Fetch Wikipedia Plots
    ↓ [adds plot text for matching]
Re-rank by Relevance
    ↓ [includes plot keyword matching]
Filter for Diversity
    ↓
Show Top 10 Results
```

---

## 📂 Key Files

| File | Purpose | Key Method |
|------|---------|-----------|
| `langgraph_tools/tools.py` | Query analysis + TMDb search | `query_intent_classifier()`, `intelligent_search_tmdb()` |
| `langgraph_tools/graph.py` | Workflow orchestration | 5-node LangGraph pipeline |
| `data_sources/tmdb_client.py` | TMDb API wrapper | `search_movies()`, `discover_movies()` |
| `data_sources/wikipedia_client.py` | Wikipedia fetcher | `get_movie_info()` |
| `vector_store/faiss_store.py` | Semantic search | `search()`, `add_movies()` |
| `retrieval/bm25_retriever.py` | Keyword search | `search()`, `build_index()` |
| `retrieval/reranker.py` | Scoring & ranking | `calculate_composite_score()`, `_calculate_wiki_plot_match()` |
| `retrieval/hybrid_retriever.py` | FAISS + BM25 fusion | `search()` |
| `main.py` | Entry point | `run_query()` |

---

## 🧠 Understanding Query Analysis

### What the LLM Extracts

```python
Query: "Dark sci-fi thriller like Blade Runner"

Analysis:
{
    "intent": "similar_to",
    "genres": ["Science Fiction", "Thriller"],
    "mood": "dark, dystopian, unsettling",
    "themes": ["artificial intelligence", "humanity", "dystopia"],
    "keywords": ["dark", "sci-fi", "thriller", "blade", "runner"],
    "mentioned_movies": ["Blade Runner"]
}
```

### Why This Matters

- **Genres**: Used for TMDb /discover filter
- **Mood**: Added to search query for enrichment
- **Themes**: Used for semantic search
- **Keywords**: Used for BM25 keyword search
- **Mentioned Movies**: Could find similar films

---

## 🔍 TMDb Search Strategies

| Strategy | When | Example |
|----------|------|---------|
| **1. Discover** | Has genres or temporal filter | `with_genres=10752,28` (War, Action) |
| **2. Enriched Search** | Always | `query=war military patriotism tense` |
| **3. Original Query** | Always | `query=war movie about indian soldiers` |
| **4. Keywords** | If < 20 results | Individual keyword searches |

---

## 📊 Re-ranking Score Breakdown

```
Base Scores (individual components):

semantic_score         = FAISS similarity        (0.0-1.0)
genre_score           = Genre overlap           (0.0-1.0)
rating_score          = vote_average / 10       (0.0-1.0)
recency_score         = exp(-age/10)            (0.0-1.0)
popularity_score      = log(popularity)/3       (0.0-1.0)
keyword_match_score   = Title + overview match  (0.0-1.0)
wiki_plot_score       = Plot + themes match     (0.0-1.0)

Final Composite:
= 0.25×semantic + 0.20×genre + 0.20×rating + 0.15×recency 
  + 0.10×popularity + 0.10×keyword + 0.05×wiki_plot

Bonus:
IF wiki_plot_score > 0.7:
    Final *= 1.1  (10% boost)
```

---

## 🎬 Example: "War movie about indian soldiers"

### Step 1: Query Analysis
```
Intent: exploration
Genres: [War, Action, Drama]
Mood: tense, patriotic
Themes: [war, military, patriotism, soldiers]
Keywords: [war, soldiers, indian]
```

### Step 2: TMDb Search Results
```
- Lagaan (2001) - Partition era India
- Hey Ram (2000) - Partition violence
- 1947 Earth (1998) - Independence context
- Raees (2017) - Modern India
- Saving Private Ryan (1998) - WWII (less relevant)
```

### Step 3: Wikipedia Enrichment
```
Lagaan plot includes:
- "indian farmers"
- "resistance against British"
- "soldiers"

Saving Private Ryan plot includes:
- "American soldiers"
- "WWII"
- NO mention of "indian"
```

### Step 4: Re-ranking
```
Lagaan:           0.57 (matches indian + soldiers in plot)
Hey Ram:          0.55 (matches partition context)
Saving Private:   0.52 (generic war match, no "indian")
```

### Step 5: Output
```
1. Lagaan (2001) ⭐⭐⭐⭐⭐
   "Excellent match: Indian independence era, soldiers, 
    patriotic themes match your query perfectly"

2. Hey Ram (2000) ⭐⭐⭐⭐
   "Good match: Focuses on Indian partition violence"
```

---

## 🔧 Configuration

Edit `config.py` to adjust:

```python
# Scoring weights
SEMANTIC_WEIGHT = 0.25      # Increase for more semantic matching
GENRE_WEIGHT = 0.20         # Increase for genre-focused search
RATING_WEIGHT = 0.20        # Increase to prefer highly-rated films
RECENCY_WEIGHT = 0.15       # Increase to prefer newer films
POPULARITY_WEIGHT = 0.10    # Increase to prefer popular films
KEYWORD_WEIGHT = 0.10       # Increase for exact keyword matching
WIKI_PLOT_WEIGHT = 0.05     # Increase for plot-based matching

# Search filters
MIN_RATING_THRESHOLD = 5.5  # Minimum rating to include
MIN_VOTE_COUNT = 10         # Minimum number of votes
DIVERSITY_K = 10            # Number of diverse results
```

---

## 🚀 Running the System

### Interactive Mode
```bash
cd backend
python main.py

# Then type queries:
>>> "War movies from the 90s"
>>> "Recommend uplifting family films"
>>> "Dark thrillers like Inception"
```

### Single Query
```bash
python main.py "Dark sci-fi thriller like Blade Runner"
```

### Batch Processing
```bash
python initialize_data.py  # Populate vector store first
python main.py  # Start interactive mode
```

---

## 📈 Performance Tips

1. **First Query Slow**: Wikipedia fetching adds ~2-3 seconds
   - Subsequent queries use cache: ~1 second

2. **Improve Relevance**: Be specific
   - ❌ "Good movies"
   - ✅ "Indian war films from 2000s"

3. **Too Many Results**: Use diversity filter
   - Set `DIVERSITY_K = 5` for fewer results

4. **Better Matches**: Provide context
   - ❌ "Recommend movies"
   - ✅ "Movies like Lagaan with patriotic themes"

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **0 results** | Check internet connection, verify TMDb API key |
| **Slow response** | Wikipedia requests slow, retry in 10s |
| **Wrong genre** | Try specifying genre explicitly |
| **Poor matches** | Add more specific keywords to query |
| **Out of memory** | Reduce `DIVERSITY_K` or `max_results` |

---

## 📚 Learning Path

1. **Start**: Read this file
2. **Understand**: Read `ARCHITECTURE.md`
3. **Deep Dive**: Read `PROCESSING_FLOW.md`
4. **Code**: Read `langgraph_tools/graph.py`
5. **Extend**: Modify `retrieval/reranker.py`

---

## 🎯 Key Concepts

| Concept | Explanation |
|---------|-------------|
| **RAG** | Retrieval-Augmented Generation: Local vector store + API fallback |
| **Hybrid Search** | Combine semantic (FAISS) + keyword (BM25) search |
| **Composite Score** | Weighted sum of multiple ranking factors |
| **MMR** | Maximal Marginal Relevance: Balance relevance + diversity |
| **Plot Matching** | Match query keywords against Wikipedia plot text |

---

## 🔗 External APIs

| API | Purpose | Rate Limit | Key Location |
|-----|---------|-----------|--------------|
| **TMDb** | Movie data | 40/10s | `.env`: `TMDB_API_KEY` |
| **Wikipedia** | Plot info | Generous | Free, no key needed |
| **Gemini** | LLM analysis | 1000/day | `.env`: `GOOGLE_API_KEY` |

---

## 💡 Next Steps

1. ✅ System is fully functional
2. 📊 Run example queries and evaluate
3. 🔧 Tune weights in `config.py`
4. 📚 Add more retrieval strategies
5. 🎯 Deploy to production

---

**Version**: 2.0 (With Wikipedia Plot Matching)  
**Last Updated**: October 31, 2025  
**Status**: Production Ready ✅
