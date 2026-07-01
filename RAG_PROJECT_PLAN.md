# 🎯 FIFA World Cup RAG System - Project Plan

---

## 📊 Project Overview

**Goal:** Build a Local RAG System that retrieves information from FIFA World Cup document using Groq API

**Stack:**
- Backend: FastAPI
- Embedding: sentence-transformers (all-MiniLM-L6-v2)
- Vector DB: FAISS (local, in-memory)
- LLM: Groq (mixtral-8x7b-32768)
- Deployment: Render (free tier)
- Testing: Postman

**Cost:** $0 (all free)

---

## 🏗️ Architecture

```
FIFA_World_Cup_Professional_Guide.docx (Input Document)
                    ↓
            [Text Extraction]
                    ↓
            [Chunking - 500 words]
                    ↓
            [Embedding - all-MiniLM-L6-v2]
                    ↓
            [FAISS Index - Vector DB]
                    ↓
    ┌───────────────────────────────┐
    │     FastAPI Backend (Render)   │
    ├───────────────────────────────┤
    │ Query → Retrieval → Generation │
    │                               │
    │ 1. User Query                 │
    │ 2. Embed Query                │
    │ 3. Search FAISS (top-5)       │
    │ 4. Send to Groq + Context     │
    │ 5. Return Answer + Sources    │
    └───────────────────────────────┘
              ↓
         Postman (Testing)
```

---

## 📂 File Structure

```
rag-world-cup/
│
├── data/
│   ├── FIFA_World_Cup_Professional_Guide.docx  ✅ (موجود)
│   ├── text_extracted.txt                       (هنعمله)
│   ├── chunks.json                              (هنعمله)
│   ├── embeddings.npy                           (هنعمله)
│   └── faiss_index.bin                          (هنعمله)
│
├── src/
│   ├── __init__.py
│   ├── chunker.py                    (تقسيم النص)
│   ├── embedder.py                   (embedding locally)
│   ├── retriever.py                  (FAISS search)
│   ├── groq_client.py               (Groq API wrapper)
│   └── rag_pipeline.py              (combine everything)
│
├── app.py                            (FastAPI main)
├── requirements.txt                  (dependencies)
├── .env                             (GROQ_API_KEY)
├── .gitignore
├── Dockerfile                       (for Render)
├── render.yaml                      (Render deployment config)
└── README.md
```

---

## 🔧 Technologies & Models

| Component | Technology | Details |
|-----------|-----------|---------|
| **Embedding** | sentence-transformers | `all-MiniLM-L6-v2` (384 dims, 22MB) |
| **Vector DB** | FAISS | In-memory, local, super fast |
| **LLM** | Groq API | `mixtral-8x7b-32768` (free with rate limits) |
| **Backend** | FastAPI | Async, auto-docs, type hints |
| **Framework** | Python 3.9+ | - |
| **Testing** | Postman | HTTP client |
| **Deployment** | Render | Free tier (750 hrs/month) |

---

## 📋 Processing Pipeline

### Phase 1: Document Processing (One-time)
```
1. Extract text from DOCX
   ├── Input: FIFA_World_Cup_Professional_Guide.docx
   └── Output: text_extracted.txt

2. Chunk the text
   ├── Strategy: Sliding window (500 words)
   ├── Overlap: 100 words (20%)
   ├── Preserve headings
   └── Output: chunks.json

3. Generate embeddings
   ├── Model: all-MiniLM-L6-v2
   ├── Batch process chunks
   ├── Save vectors
   └── Output: embeddings.npy

4. Build FAISS index
   ├── Load embeddings
   ├── Create FAISS index
   ├── Save index
   └── Output: faiss_index.bin
```

### Phase 2: Runtime (Every Query)
```
1. Receive query from user (Postman)
2. Embed the query (same model)
3. Search FAISS (top-5 most similar chunks)
4. Format context + prompt
5. Send to Groq API
6. Return answer + source chunks
```

---

## 🔌 FastAPI Endpoints

### Endpoint 1: Health Check
```
GET /api/health

Response:
{
  "status": "ok",
  "version": "1.0.0",
  "model": "mixtral-8x7b-32768"
}
```

### Endpoint 2: Query (Main)
```
POST /api/query

Request:
{
  "question": "Who won the 2022 World Cup?",
  "top_k": 5,
  "temperature": 0.7,
  "max_tokens": 500
}

Response:
{
  "question": "Who won the 2022 World Cup?",
  "answer": "Argentina won the 2022 FIFA World Cup...",
  "sources": [
    {
      "chunk_id": 0,
      "text": "...",
      "relevance_score": 0.95
    },
    {
      "chunk_id": 1,
      "text": "...",
      "relevance_score": 0.88
    }
  ],
  "retrieval_time_ms": 45,
  "generation_time_ms": 1200
}
```

### Endpoint 3: Batch Evaluation (for Q&A testing)
```
POST /api/evaluate

Request:
{
  "questions": [
    {
      "id": 1,
      "question": "How many World Cups did Brazil win?",
      "expected_answer": "5"
    },
    {
      "id": 2,
      "question": "Who won in 2022?",
      "expected_answer": "Argentina"
    }
  ]
}

Response:
{
  "results": [
    {
      "id": 1,
      "question": "...",
      "retrieved_chunks": 5,
      "answer": "Brazil has won 5 World Cups...",
      "expected": "5",
      "match": true,
      "retrieval_score": 0.92
    },
    ...
  ],
  "summary": {
    "total": 45,
    "correct": 43,
    "accuracy": 0.956
  }
}
```

---

## 📝 Processing Settings

| Setting | Value | Reason |
|---------|-------|--------|
| Chunk Size | 500 words | Good balance between context & specificity |
| Chunk Overlap | 100 words (20%) | Preserve context across chunks |
| Top-K Retrieval | 5 chunks | Balance between relevance & context |
| Embedding Dim | 384 (all-MiniLM) | Fast, sufficient for domain knowledge |
| Temperature | 0.7 | Balanced between creative & factual |
| Max Tokens | 500 | Enough for detailed answers |
| Rate Limit (Groq) | 30 req/min free tier | Plan accordingly for testing |

---

## 🚀 Execution Plan

### Step 1: Setup (Today)
- [ ] Extract text from DOCX → `text_extracted.txt`
- [ ] Create chunker.py → generate `chunks.json`
- [ ] Create embedder.py → generate `embeddings.npy`
- [ ] Create retriever.py → generate `faiss_index.bin`

### Step 2: Backend Development
- [ ] Create groq_client.py (Groq API integration)
- [ ] Create rag_pipeline.py (combine retrieval + generation)
- [ ] Create app.py (FastAPI routes)
- [ ] Create requirements.txt
- [ ] Create .env template

### Step 3: Testing (Postman)
- [ ] Test /api/health endpoint
- [ ] Test /api/query with 5-10 questions
- [ ] Test /api/evaluate with all 45 Q&A

### Step 4: Deployment (Render)
- [ ] Create Dockerfile
- [ ] Create render.yaml
- [ ] Push to GitHub
- [ ] Deploy to Render
- [ ] Test live endpoints

### Step 5: Optimization
- [ ] Monitor Groq API usage
- [ ] Tweak chunk size if needed
- [ ] Add caching (optional)
- [ ] Add rate limiting (optional)

---

## 💾 Data Preparation

**Input Document:**
- File: `FIFA_World_Cup_Professional_Guide.docx` ✅ (11 pages, 4500+ words)
- Format: DOCX
- Content: World Cup history, statistics, Q&A

**Processing Output:**
1. `chunks.json` - ~20-25 chunks (500 words each)
2. `embeddings.npy` - ~20-25 vectors (384 dimensions each)
3. `faiss_index.bin` - FAISS index for fast search
4. `metadata.json` - chunk IDs, text, sources (optional)

---

## 🧪 Testing Strategy

### Unit Testing
```
✅ Chunker: Verify chunks are 500 words ± 10%
✅ Embedder: Verify embeddings are 384-dim vectors
✅ Retriever: Verify FAISS returns top-5 relevant chunks
✅ Groq Client: Verify API calls work correctly
```

### Integration Testing (Postman)
```
✅ /api/health returns 200
✅ /api/query returns answer + sources
✅ /api/evaluate returns accuracy metrics
```

### Evaluation Testing (45 Q&A)
```
✅ Run all 45 questions from document
✅ Measure retrieval accuracy (chunk relevance)
✅ Measure generation accuracy (answer correctness)
✅ Calculate overall RAG metrics
```

---

## 📊 Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Retrieval Recall@5 | >85% | Top-5 chunks contain answer |
| Generation Accuracy | >80% | Groq returns correct answer |
| Query Latency | <2s | End-to-end response time |
| Uptime | >95% | Render free tier availability |
| Q&A Accuracy | >85% | Correct answers on 45 test questions |

---

## 🔐 Environment Variables

```
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=mixtral-8x7b-32768
MAX_TOKENS=500
TEMPERATURE=0.7
TOP_K=5
```

---

## 📚 Dependencies (requirements.txt)

```
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
groq==0.4.2
sentence-transformers==2.2.2
faiss-cpu==1.7.4
numpy==1.24.3
python-docx==0.8.11
pydantic==2.0.0
```

---

## 🎯 Next Steps

1. **Ready?** Confirm this plan
2. **Start Phase 1:** Text extraction + chunking
3. **Build embeddings:** Generate FAISS index
4. **Develop FastAPI:** Create routes
5. **Test in Postman:** Validate endpoints
6. **Deploy on Render:** Go live

---

## 📞 Questions to Confirm

- [ ] Chunk size 500 words OK?
- [ ] Top-K = 5 chunks OK?
- [ ] Temperature 0.7 OK?
- [ ] Do we need caching?
- [ ] Do we need rate limiting?
- [ ] Do we need authentication?

---

**Status:** ✅ Ready to Start Phase 1

**Date:** June 28, 2024
