#!/usr/bin/env python3
"""
Generate embeddings for all chunks using sentence-transformers
Model: all-MiniLM-L6-v2 (384 dimensions, 22MB)
"""

import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from pathlib import Path

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"📥 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embeddings = []
        self.chunks_data = []
    
    def load_chunks(self, chunks_json_path: str):
        """Load chunks from JSON"""
        with open(chunks_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks_data = data.get("chunks", [])
        print(f"✅ Loaded {len(self.chunks_data)} chunks")
        
        return self.chunks_data
    
    def generate_embeddings(self, batch_size=8):
        """Generate embeddings for all chunks"""
        if not self.chunks_data:
            print("❌ No chunks to embed!")
            return
        
        print(f"\n🔄 Generating embeddings ({len(self.chunks_data)} chunks, batch_size={batch_size})...")
        
        texts = [chunk["text"] for chunk in self.chunks_data]
        
        # Generate embeddings (sentence-transformers handles batching)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.embeddings = embeddings
        
        print(f"\n✅ Generated embeddings:")
        print(f"   - Shape: {embeddings.shape}")
        print(f"   - Type: {embeddings.dtype}")
        print(f"   - Memory: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        
        return embeddings
    
    def save_embeddings(self, output_dir: str):
        """Save embeddings and metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings as numpy
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_path, self.embeddings)
        print(f"💾 Saved embeddings: {embeddings_path}")
        
        # Save metadata (mapping chunk_id → embedding index)
        metadata = {
            "model": self.model_name,
            "embedding_dim": int(self.embeddings.shape[1]),
            "total_chunks": len(self.chunks_data),
            "chunks": [
                {
                    "id": chunk["id"],
                    "type": chunk["type"],
                    "word_count": chunk["word_count"],
                    "source_section": chunk.get("source_section", ""),
                    "embedding_index": idx
                }
                for idx, chunk in enumerate(self.chunks_data)
            ]
        }
        
        metadata_path = os.path.join(output_dir, "embeddings_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved metadata: {metadata_path}")
        
        return embeddings_path, metadata_path
    
    def get_embedding_stats(self):
        """Print embedding statistics"""
        if len(self.embeddings) == 0:
            print("❌ No embeddings generated yet!")
            return
        
        # Calculate norms
        norms = np.linalg.norm(self.embeddings, axis=1)
        
        # Similarity stats
        similarity_matrix = self.embeddings @ self.embeddings.T
        
        print(f"""
📊 Embedding Statistics:
   - Total Embeddings: {len(self.embeddings)}
   - Dimension: {self.embeddings.shape[1]}
   - Dtype: {self.embeddings.dtype}
   
🔍 Norm Statistics:
   - Mean Norm: {norms.mean():.4f}
   - Min Norm: {norms.min():.4f}
   - Max Norm: {norms.max():.4f}
   
📈 Similarity Statistics (Self-Similarity Matrix):
   - Diagonal (self-similarity): Mean = {np.diag(similarity_matrix).mean():.4f}
   - Off-diagonal (cross-similarity): Mean = {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean():.4f}
   - Off-diagonal (cross-similarity): Max = {similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].max():.4f}
        """)
    
    def print_sample_embeddings(self, num_samples=3):
        """Print sample embeddings"""
        if len(self.embeddings) == 0:
            return
        
        print(f"\n📋 Sample Embeddings (first {num_samples} chunks):")
        for i in range(min(num_samples, len(self.embeddings))):
            chunk = self.chunks_data[i]
            emb = self.embeddings[i]
            
            print(f"\n   Chunk {chunk['id']} ({chunk['type'].upper()}):")
            print(f"   - Text: {chunk['text'][:80].replace(chr(10), ' ')}...")
            print(f"   - Embedding (first 5): {emb[:5]}")
            print(f"   - Norm: {np.linalg.norm(emb):.4f}")

def main():
    chunks_path = "./data/chunks.json"
    output_dir = "./data"
    
    if not os.path.exists(chunks_path):
        print(f"❌ Error: {chunks_path} not found")
        return
    
    print("🚀 Starting Embedding Generation...\n")
    
    # Initialize generator
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    # Load chunks
    generator.load_chunks(chunks_path)
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(batch_size=8)
    
    # Print statistics
    generator.get_embedding_stats()
    
    # Print samples
    generator.print_sample_embeddings(num_samples=3)
    
    # Save embeddings
    print(f"\n💾 Saving to {output_dir}...")
    generator.save_embeddings(output_dir)
    
    print(f"""
✅ Embedding Generation Complete!

📊 Summary:
   - Model: all-MiniLM-L6-v2
   - Chunks Embedded: {len(generator.chunks_data)}
   - Dimension: 384
   - Output Files:
     • embeddings.npy
     • embeddings_metadata.json
    """)

if __name__ == "__main__":
    main()
