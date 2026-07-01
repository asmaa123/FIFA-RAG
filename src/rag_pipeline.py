#!/usr/bin/env python3
"""
RAG Pipeline - Retrieval + Augmented + Generation
"""

import time
from typing import List, Dict
from src.retriever import FAISSRetriever
from src.groq_client import GroqClient

class RAGPipeline:
    def __init__(self, 
                 chunks_json: str,
                 embeddings_npy: str,
                 groq_api_key: str = None,
                 top_k: int = 5):
        """Initialize RAG pipeline"""
        
        print("🚀 Initializing RAG Pipeline...\n")
        
        # Initialize retriever
        print("1️⃣  Initializing Retriever...")
        self.retriever = FAISSRetriever(chunks_json, embeddings_npy)
        self.top_k = top_k
        
        # Initialize LLM
        print("\n2️⃣  Initializing Groq Client...")
        self.groq = GroqClient(api_key=groq_api_key)
        
        print("\n✅ RAG Pipeline Ready!\n")
    
    def query(self, question: str, top_k: int = None) -> Dict:
        """Process a query end-to-end"""
        
        if top_k is None:
            top_k = self.top_k
        
        start_time = time.time()
        
        # Step 1: Retrieve
        print(f"🔍 Retrieving relevant chunks...")
        retrieval_start = time.time()
        
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        
        retrieval_time = (time.time() - retrieval_start) * 1000  # ms
        print(f"   ✅ Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.0f}ms")
        
        # Step 2: Generate
        print(f"\n🤖 Generating answer using Groq...")
        generation_start = time.time()
        
        result = self.groq.generate_with_sources(question, retrieved_chunks)
        
        generation_time = (time.time() - generation_start) * 1000  # ms
        print(f"   ✅ Generated answer in {generation_time:.0f}ms")
        
        # Add timing info
        total_time = (time.time() - start_time) * 1000
        
        result["metrics"] = {
            "retrieval_time_ms": retrieval_time,
            "generation_time_ms": generation_time,
            "total_time_ms": total_time,
            "chunks_retrieved": len(retrieved_chunks)
        }
        
        return result
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        """Process multiple questions"""
        results = []
        
        print(f"📝 Processing {len(questions)} questions...\n")
        
        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {question}")
            result = self.query(question)
            results.append(result)
            print(f"   ✅ Done\n")
        
        return results
    
    def evaluate_qa(self, qa_pairs: List[Dict]) -> Dict:
        """Evaluate on Q&A pairs with metrics"""
        
        print(f"📊 Evaluating on {len(qa_pairs)} Q&A pairs...\n")
        
        results = []
        correct = 0
        
        for qa in qa_pairs:
            question = qa["question"]
            expected_answer = qa.get("answer", "")
            
            # Query
            result = self.query(question, top_k=5)
            
            # Simple check: does answer contain key parts of expected answer?
            # (in production, use proper evaluation metrics)
            answer_lower = result["answer"].lower()
            expected_lower = expected_answer.lower()
            
            # Check for key keywords
            words = expected_lower.split()
            matched_words = sum(1 for word in words if len(word) > 3 and word in answer_lower)
            match_ratio = matched_words / len([w for w in words if len(w) > 3]) if words else 0
            
            is_correct = match_ratio > 0.5  # At least 50% keyword match
            if is_correct:
                correct += 1
            
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": result["answer"],
                "match_ratio": match_ratio,
                "is_correct": is_correct,
                "sources_used": len(result["sources"])
            })
        
        accuracy = correct / len(qa_pairs) if qa_pairs else 0
        
        return {
            "total_questions": len(qa_pairs),
            "correct_answers": correct,
            "accuracy": accuracy,
            "results": results
        }

def main():
    """Test RAG pipeline"""
    
    chunks_path = "./data/chunks.json"
    embeddings_path = "./data/embeddings.npy"
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(chunks_path, embeddings_path, top_k=5)
        
        # Test queries
        test_questions = [
            "Who won the 2022 World Cup?",
            "How many times did Brazil win the World Cup?",
            "What are the records in World Cup history?"
        ]
        
        print("=" * 80)
        print("🧪 Testing RAG Pipeline")
        print("=" * 80)
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            print("-" * 80)
            
            result = pipeline.query(question, top_k=3)
            
            print(f"\n💡 Answer:\n{result['answer']}\n")
            
            print(f"📚 Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"   [{i}] {source['source_section']} (Similarity: {source['similarity']:.3f})")
            
            print(f"\n⏱️  Metrics:")
            print(f"   - Retrieval: {result['metrics']['retrieval_time_ms']:.0f}ms")
            print(f"   - Generation: {result['metrics']['generation_time_ms']:.0f}ms")
            print(f"   - Total: {result['metrics']['total_time_ms']:.0f}ms")
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   → Make sure embeddings are generated and GROQ_API_KEY is set")

if __name__ == "__main__":
    main()
