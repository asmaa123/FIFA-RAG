# 🎯 FIFA World Cup RAG System

A **Retrieval-Augmented Generation (RAG)** system for FIFA World Cup knowledge retrieval and question answering.

**Stack:** FastAPI + sentence-transformers + FAISS + Groq API

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- Groq API Key ([Get it free](https://console.groq.com))

### 2. Setup

```bash
# Clone/download the project
cd rag-world-cup

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Prepare Data

```bash
# Extract text from DOCX
python extract_docx.py

# Generate chunks
python src/chunker.py

# Generate embeddings (one-time, ~2-3 minutes)
python src/embedder.py
```

### 4. Run Server

```bash
# Local development
python -m uvicorn app:app --reload

# Production
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Server will be available at `http://localhost:8000`

---

## 📚 API Endpoints

### 1. Health Check
```
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "mixtral-8x7b-32768",
  "chunks_loaded": 8
}
```

### 2. Query (Main RAG Endpoint)
```
POST /api/query
Content-Type: application/json

{
  "question": "Who won the 2022 World Cup?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "Who won the 2022 World Cup?",
  "answer": "Argentina won the 2022 FIFA World Cup...",
  "sources": [
    {
      "chunk_id": 0,
      "source_section": "Q&A Section",
      "similarity": 0.95,
      "text_preview": "Argentina won the 2022 World Cup..."
    }
  ],
  "metrics": {
    "retrieval_time_ms": 45,
    "generation_time_ms": 1200,
    "total_time_ms": 1245,
    "chunks_retrieved": 5
  }
}
```

### 3. Batch Query
```
POST /api/batch-query
Content-Type: application/json

[
  "Who won the 2022 World Cup?",
  "How many times did Brazil win?",
  "What are the World Cup records?"
]
```

### 4. Evaluate
```
POST /api/evaluate
Content-Type: application/json

{
  "questions": [
    {
      "question": "Who won the 2022 World Cup?",
      "answer": "Argentina"
    },
    {
      "question": "How many World Cups did Brazil win?",
      "answer": "5"
    }
  ]
}
```

---

## 🧪 Testing with Postman

1. **Import Collection** (optional)
2. **Set Base URL:** `http://localhost:8000`
3. **Test Endpoints:**

```
GET http://localhost:8000/api/health

POST http://localhost:8000/api/query
{
  "question": "Who won the 2022 World Cup?",
  "top_k": 5
}

POST http://localhost:8000/api/batch-query
["Who won 2022?", "How many times Brazil?"]
```

---

## 📂 Project Structure

```
rag-world-cup/
├── app.py                    # FastAPI main application
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── Dockerfile               # Docker configuration
├── render.yaml              # Render deployment config
│
├── extract_docx.py          # Extract text from DOCX
├── src/
│   ├── chunker.py          # Text chunking (500-word chunks)
│   ├── embedder.py         # Generate embeddings
│   ├── retriever.py        # FAISS semantic search
│   ├── groq_client.py      # Groq API wrapper
│   └── rag_pipeline.py     # RAG pipeline (retrieval + generation)
│
├── data/
│   ├── FIFA_World_Cup_Professional_Guide.docx  # Source document
│   ├── text_extracted.txt                       # Extracted text
│   ├── chunks.json                              # Chunks metadata
│   ├── embeddings.npy                           # Embeddings (384-dim)
│   └── embeddings_metadata.json                 # Embedding metadata
│
└── README.md                # This file
```

---

## 🔧 Configuration

### Environment Variables

```
# Required
GROQ_API_KEY=your_api_key_here

# Optional (defaults below)
HOST=0.0.0.0
PORT=8000
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile
TOP_K=5
MAX_TOKENS=500
TEMPERATURE=0.7
```

---

## 📊 RAG Pipeline Flow

```
User Question
    ↓
[Query Embedding] (sentence-transformers)
    ↓
[FAISS Search] (Find top-5 similar chunks)
    ↓
[Retrieved Context] (5 chunks with similarity scores)
    ↓
[Groq API] (mixtral-8x7b-32768 + context)
    ↓
[Generated Answer] (With source attribution)
    ↓
User Response
```

---

## 🎯 Performance Metrics

| Metric | Value |
|--------|-------|
| Embedding Model | all-MiniLM-L6-v2 (384-dim, 22MB) |
| Retrieval Speed | <100ms (FAISS) |
| Generation Speed | ~1-2s (Groq API) |
| Total Latency | ~1-2.5s per query |
| Model | mixtral-8x7b-32768 |
| Context Window | 32K tokens |

---

## 🚀 Deployment on Render

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial RAG system"
git push origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com)
2. Connect GitHub repository
3. Create new "Web Service"
4. Select this repository
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `uvicorn app:app --host 0.0.0.0 --port 8000`
7. Add environment variable: `GROQ_API_KEY=your_key`
8. Deploy!

### Step 3: Test Live
```bash
curl https://your-service.onrender.com/api/health
```

---

## 📈 Chunking Strategy

**Hybrid Approach:**
- **Narrative Text:** 500-word chunks with 100-word overlap
- **Tables:** 1 chunk per table (preserved structure)
- **Q&A:** 1 chunk per question-answer pair

**Result:**
- ~60-70 total chunks
- Balanced context length
- Preserved semantic units

---

## 🧠 Models Used

### Embedding Model
- **Name:** all-MiniLM-L6-v2
- **Dimensions:** 384
- **Speed:** ⚡⚡⚡ (Very fast)
- **Quality:** ⭐⭐⭐ (Good for domain knowledge)
- **Size:** 22MB

### LLM Model
- **Provider:** Groq
- **Model:** llama-3.3-70b-versatile
- **Type:** MoE (Mixture of Experts)
- **Speed:** ⚡⚡⚡ (Super fast)
- **Quality:** ⭐⭐⭐⭐ (Excellent)
- **Context:** 32K tokens

---

## 🧪 Evaluation

The system includes 45 Q&A pairs for evaluation:

```bash
# Test on Q&A dataset
python -c "
from src.rag_pipeline import RAGPipeline
import json

with open('data/chunks.json') as f:
    chunks = json.load(f)

# Extract Q&A from chunks
qa_pairs = [c for c in chunks['chunks'] if c['type'] == 'qa']

pipeline = RAGPipeline('data/chunks.json', 'data/embeddings.npy')
results = pipeline.evaluate_qa(qa_pairs)

print(f'Accuracy: {results[\"accuracy\"]:.1%}')
"
```

---

## 🔒 Rate Limiting

**Groq Free Tier:**
- 30 requests/minute
- Recommended: Add rate limiting for production

---

## 🐛 Troubleshooting

### Issue: "GROQ_API_KEY not found"
**Solution:** Set environment variable
```bash
export GROQ_API_KEY=your_key_here  # macOS/Linux
set GROQ_API_KEY=your_key_here     # Windows
```

### Issue: "embeddings.npy not found"
**Solution:** Generate embeddings first
```bash
python src/embedder.py
```

### Issue: "Module not found: sentence_transformers"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Slow embedding generation
**Solution:** Normal (2-3 minutes for first run due to model download)

---

## 📝 License

MIT License - Free to use and modify

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Better Q&A extraction
- Advanced retrieval (hybrid BM25 + semantic)
- Evaluation metrics (BLEU, ROUGE)
- Caching layer (Redis)
- Rate limiting (slowapi)

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Verify .env configuration
3. Check logs: `uvicorn app:app --log-level debug`

---

**Built with ❤️ for FIFA World Cup enthusiasts**
