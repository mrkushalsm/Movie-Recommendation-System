# Movie Recommendation System with RAG

Advanced movie recommendation system using **LangGraph + RAG (Retrieval Augmented Generation)**.

## Architecture

**RAG Components:**
- **Vector Store**: FAISS with semantic embeddings (paraphrase-MiniLM-L3-v2)
- **BM25**: Sparse retrieval for keyword matching
- **Hybrid Retrieval**: Combines both approaches
- **Re-ranking**: 5-factor scoring (semantic, genre, rating, recency, popularity)
- **LLM**: Google Gemini 1.5 Flash for explanations

**Data Sources:**
- Local FAISS vector store (~1000 movies)
- TMDb API (fallback for new movies)
- Wikipedia (enrichment)
- Redis cache (optional)

## Setup

### 1. Install Dependencies
```bash
conda create -n movie-rec python=3.10
conda activate movie-rec
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
# Edit .env with your keys:
# - TMDB_API_KEY=your_key
# - GOOGLE_API_KEY=your_key
```

### 3. Initialize RAG Database
```bash
python initialize_data.py
```
This fetches ~1000 movies from TMDb and builds the vector store + BM25 index.
**Takes 10-30 minutes** depending on network speed.

### 4. Optional: Start Redis
```bash
docker run -d -p 6379:6379 redis:latest
```

## Usage

```bash
python main.py
```

Example queries:
- "Best sci-fi movies from 2020"
- "War movies about Indian soldiers"
- "Classic thrillers from the 80s"

## Current Issue

**Network connectivity problem**: Your system cannot reach TMDb API.

```
Error: Connection aborted, ConnectionResetError(104, 'Connection reset by peer')
```

**Solutions:**
1. Check firewall/proxy settings
2. Try different network (mobile hotspot, VPN)
3. Wait and retry (TMDb API might be temporarily down)
4. Once network is fixed, run `python initialize_data.py` to build RAG database

**Without RAG database**, system falls back to pure API mode (which currently fails due to network).

**With RAG database**, system works offline using local vector store!
