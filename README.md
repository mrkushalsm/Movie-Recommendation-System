# Movie Recommendation System with RAG# Movie Recommendation System with RAG



A production-ready movie recommendation system using LangGraph, Gemini 2.0 Flash LLM, and hybrid RAG (Retrieval-Augmented Generation).Advanced movie recommendation system using **LangGraph + RAG (Retrieval Augmented Generation)**.



## Project Structure## Architecture



```**RAG Components:**

.- **Vector Store**: FAISS with semantic embeddings (paraphrase-MiniLM-L3-v2)

├── backend/                          # Backend services and Python code- **BM25**: Sparse retrieval for keyword matching

│   ├── main.py                      # Main application entry point- **Hybrid Retrieval**: Combines both approaches

│   ├── config.py                    # Configuration settings- **Re-ranking**: 5-factor scoring (semantic, genre, rating, recency, popularity)

│   ├── requirements.txt             # Python dependencies- **LLM**: Google Gemini 1.5 Flash for explanations

│   ├── README.md                    # Backend documentation

│   ├── .env                         # Environment variables (API keys)**Data Sources:**

│   ├── .env.example                 # Example environment file- Local FAISS vector store (~1000 movies)

│   ├── initialize_data.py           # Database initialization- TMDb API (fallback for new movies)

│   ├── movie_recommendation_colab.ipynb  # Google Colab notebook version- Wikipedia (enrichment)

│   ├── data/                        # Cache and vector store data- Redis cache (optional)

│   ├── data_sources/                # API clients (TMDb, Wikipedia, Ratings)

│   ├── langgraph_tools/             # LangGraph workflow and tools## Setup

│   ├── retrieval/                   # Retrieval and re-ranking logic

│   ├── utils/                       # Utility functions### 1. Install Dependencies

│   └── vector_store/                # Vector store and embeddings```bash

│conda create -n movie-rec python=3.10

└── frontend/                         # Frontend application (empty, ready for setup)conda activate movie-rec

    └── README.md                    # Frontend documentationpip install -r requirements.txt

```

.gitignore                           # Git ignore rules for entire project

LICENSE                              # Project license### 2. Configure API Keys

README.md                            # This file```bash

```cp .env.example .env

# Edit .env with your keys:

## Quick Start# - TMDB_API_KEY=your_key

# - GOOGLE_API_KEY=your_key

### Backend Setup```



```bash### 3. Initialize RAG Database

cd backend```bash

pip install -r requirements.txtpython initialize_data.py

cp .env.example .env```

# Edit .env with your API keysThis fetches ~1000 movies from TMDb and builds the vector store + BM25 index.

python main.py**Takes 10-30 minutes** depending on network speed.

```

### 4. Optional: Start Redis

### Frontend Setup```bash

docker run -d -p 6379:6379 redis:latest

Coming soon! Add your React, Vue, or Next.js application in the `frontend/` folder.```



## Features## Usage



- 🧠 **LLM-Powered Analysis**: Gemini 2.0 Flash for intelligent query understanding```bash

- 🔍 **Hybrid RAG**: FAISS vector store + BM25 keyword searchpython main.py

- 🎯 **Multi-Strategy Search**: Combines multiple search strategies for better results```

- ⭐ **Intelligent Re-ranking**: Composite scoring with semantic, rating, recency, and keyword matching

- 💾 **Dynamic Database**: Grows with each queryExample queries:

- 📊 **Production Ready**: Designed for scalable deployment- "Best sci-fi movies from 2020"

- "War movies about Indian soldiers"

## Technology Stack- "Classic thrillers from the 80s"



### Backend## Current Issue

- **LLM**: Google Gemini 2.0 Flash

- **Vector Store**: FAISS with Sentence Transformers**Network connectivity problem**: Your system cannot reach TMDb API.

- **Keyword Search**: BM25

- **Workflow**: LangGraph```

- **APIs**: TMDb, WikipediaError: Connection aborted, ConnectionResetError(104, 'Connection reset by peer')

- **Framework**: Python 3.8+```



### Frontend**Solutions:**

- Ready for React, Vue, Next.js, or other frameworks1. Check firewall/proxy settings

2. Try different network (mobile hotspot, VPN)

## Documentation3. Wait and retry (TMDb API might be temporarily down)

4. Once network is fixed, run `python initialize_data.py` to build RAG database

- [Backend README](backend/README.md) - Detailed backend setup and architecture

- [Frontend README](frontend/README.md) - Frontend setup instructions**Without RAG database**, system falls back to pure API mode (which currently fails due to network).

- [Colab Notebook](backend/movie_recommendation_colab.ipynb) - Interactive notebook for Google Colab

**With RAG database**, system works offline using local vector store!

## Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.
