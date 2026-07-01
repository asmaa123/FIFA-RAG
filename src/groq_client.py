#!/usr/bin/env python3
"""
Groq API Client - Generate answers using mixtral-8x7b-32768
"""

from groq import Groq
from typing import List, Dict
import os

class GroqClient:
    def __init__(self, 
                 api_key: str = None,
                 model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.7,
                 max_tokens: int = 500):
        """Initialize Groq client"""
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY not found in environment!")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        print(f"✅ Groq Client Initialized")
        print(f"   - Model: {model}")
        print(f"   - Temperature: {temperature}")
        print(f"   - Max Tokens: {max_tokens}")
    
    def generate_answer(self, 
                       question: str, 
                       context: str,
                       system_prompt: str = None) -> str:
        """Generate answer using Groq API with context"""
        
        if system_prompt is None:
            system_prompt = """You are a knowledgeable FIFA World Cup expert. 
Answer questions based on the provided context. Be accurate, concise, and cite the context when relevant.
If the context doesn't contain enough information, say so honestly."""
        
        # Build prompt
        user_message = f"""Context:
{context}

Question: {question}

Answer based on the context above."""
        
        # Call Groq API
        message = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Extract answer
        answer = message.choices[0].message.content
        
        return answer
    
    def generate_with_sources(self,
                             question: str,
                             retrieved_chunks: List[Dict]) -> Dict:
        """Generate answer with source attribution"""
        
        # Format context from chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Source {i}] ({chunk['source_section']}):\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "source_section": chunk["source_section"],
                    "similarity": chunk.get("similarity", 0),
                    "text_preview": chunk["text"][:100]
                }
                for chunk in retrieved_chunks
            ]
        }
    
    def batch_generate(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Generate answers for multiple Q&A pairs"""
        results = []
        
        for qa in qa_pairs:
            question = qa.get("question")
            context = qa.get("context", "")
            
            answer = self.generate_answer(question, context)
            
            results.append({
                "question": question,
                "answer": answer,
                "expected_answer": qa.get("expected_answer", "")
            })
        
        return results

def main():
    """Test Groq client"""
    
    try:
        client = GroqClient()
        
        # Test question
        test_question = "Who won the 2022 FIFA World Cup?"
        test_context = """
        Argentina won the 2022 FIFA World Cup held in Qatar. 
        It was Argentina's third World Cup title, with previous victories in 1978 and 1986.
        Lionel Messi finally achieved his World Cup dream after multiple tournament appearances.
        """
        
        print(f"\n❓ Test Question: {test_question}")
        print(f"\n📝 Generating answer using Groq...\n")
        
        answer = client.generate_answer(test_question, test_context)
        
        print(f"✅ Answer:\n{answer}")
        
    except ValueError as e:
        print(f"⚠️ {e}")
        print("   → Set GROQ_API_KEY environment variable to test")

if __name__ == "__main__":
    main()
