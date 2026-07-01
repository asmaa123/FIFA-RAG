#!/usr/bin/env python3
"""
FAISS Retriever - Semantic search using embeddings
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

class FAISSRetriever:
    def __init__(self, chunks_json: str, embeddings_npy: str, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize FAISS retriever with precomputed embeddings"""
        
        print("📥 Loading chunks...")
        with open(chunks_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chunks = data.get("chunks", [])
        
        print("📥 Loading embeddings...")
        self.embeddings = np.load(embeddings_npy).astype('float32')
        
        print("🔧 Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        print("📥 Loading embedding model for queries...")
        self.model = SentenceTransformer(model_name)
        
        print(f"✅ FAISS Retriever Ready!")
        print(f"   - Chunks: {len(self.chunks)}")
        print(f"   - Embeddings: {self.embeddings.shape}")
        print(f"   - Index type: IndexFlatL2")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query"""
        
        # Embed the query
        query_embedding = self.model.encode(query, convert_to_numpy=True).astype('float32').reshape(1, -1)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            
            # Convert L2 distance to similarity score (0-1)
            # L2 distance: smaller = more similar
            # We'll convert it: similarity = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + distance)
            
            results.append({
                "chunk_id": chunk["id"],
                "type": chunk["type"],
                "text": chunk["text"],
                "source_section": chunk.get("source_section", ""),
                "word_count": chunk["word_count"],
                "distance": float(distance),
                "similarity": float(similarity)
            })
        
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> Dict[str, List[Dict]]:
        """Retrieve for multiple queries"""
        results = {}
        for query in queries:
            results[query] = self.retrieve(query, top_k)
        return results

def main():
    """Test retriever"""
    
    chunks_path = "/home/claude/data/chunks.json"
    embeddings_path = "/home/claude/data/embeddings.npy"
    
    # Initialize
    retriever = FAISSRetriever(chunks_path, embeddings_path)
    
    # Test queries
    test_queries = [
        "Who won the 2022 World Cup?",
        "How many World Cups did Brazil win?",
        "What are the records in World Cup history?",
        "Who is the top scorer?",
        "Argentina World Cup wins"
    ]
    
    print(f"\n🔍 Testing Retrieval with {len(test_queries)} queries...\n")
    
    for query in test_queries:
        print(f"❓ Query: {query}")
        results = retriever.retrieve(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"   [{i}] (Similarity: {result['similarity']:.3f}) [{result['type'].upper()}]")
            print(f"       Section: {result['source_section']}")
            print(f"       Text: {result['text'][:80].replace(chr(10), ' ')}...")
            print()

if __name__ == "__main__":
    main()
