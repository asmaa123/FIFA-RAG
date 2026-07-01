#!/usr/bin/env python3
"""
Improved Chunker - Extract Q&A and chunks correctly
Simple and clear implementation
"""

import json
import re
from typing import List, Dict

class SimpleChunker:
    def __init__(self):
        self.chunks = []
        self.chunk_id = 0
    
    def add_chunk(self, text: str, chunk_type: str, source: str = ""):
        """Add single chunk to the list"""
        if len(text.strip()) < 20:  # Skip very small chunks
            return
        
        chunk = {
            "id": self.chunk_id,
            "type": chunk_type,
            "text": text.strip(),
            "word_count": len(text.split()),
            "source_section": source
        }
        
        self.chunks.append(chunk)
        self.chunk_id += 1
    
    # ===== NARRATIVE CHUNKING =====
    def chunk_narrative(self, text: str, section_name: str = ""):
        """Split narrative text into 500-word chunks"""
        words = text.split()
        
        if len(words) < 500:
            self.add_chunk(text, "narrative", section_name)
            return
        
        # Split into 500-word chunks with overlap
        for i in range(0, len(words), 400):  # 400 = overlap
            chunk_words = words[i:i+500]
            if chunk_words:
                chunk_text = " ".join(chunk_words)
                self.add_chunk(chunk_text, "narrative", section_name)
    
    # ===== TABLE CHUNKING =====
    def chunk_table(self, rows: List[List[str]], table_name: str = ""):
        """Create single chunk for entire table"""
        if not rows:
            return
        
        # Format table nicely
        lines = []
        
        # Header
        if rows:
            header = " | ".join(rows[0])
            lines.append(header)
            lines.append("-" * min(len(header), 100))
        
        # Data rows
        for row in rows[1:]:
            lines.append(" | ".join(row))
        
        table_text = "\n".join(lines)
        self.add_chunk(table_text, "table", table_name)
    
    # ===== Q&A CHUNKING =====
    def chunk_qa(self, qa_list: List[Dict]):
        """Create chunk for each question-answer pair"""
        for qa in qa_list:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
            
            if question and answer:
                # Combine Q&A
                qa_text = f"Q: {question}\n\nA: {answer}"
                self.add_chunk(qa_text, "qa", f"Q&A: {question[:40]}...")
    
    # ===== MAIN PROCESSING =====
    def process_document(self, extracted_json_path: str):
        """Process entire document"""
        print("📖 Loading document...")
        
        with open(extracted_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        paragraphs = data.get("paragraphs", [])
        tables = data.get("tables", [])
        
        print(f"✅ Found {len(paragraphs)} paragraphs and {len(tables)} tables")
        
        # ===== PROCESS PARAGRAPHS =====
        print("\n📝 Processing text sections...")
        current_section = "Unknown"
        narrative_buffer = []
        qa_list = []
        
        for para in paragraphs:
            text = para.get("text", "").strip()
            is_heading = para.get("is_heading", False)
            
            if is_heading:
                # Save buffered narrative
                if narrative_buffer:
                    narrative_text = " ".join(narrative_buffer)
                    self.chunk_narrative(narrative_text, current_section)
                    narrative_buffer = []
                
                # Check: Is this a question?
                if text.startswith("Q") and ":" in text:
                    # This is a question - don't change section
                    pass
                else:
                    # Regular section
                    current_section = text
                    print(f"   ➜ Section: {current_section}")
            else:
                # Regular text - add to buffer
                narrative_buffer.append(text)
        
        # Save remaining narrative
        if narrative_buffer:
            narrative_text = " ".join(narrative_buffer)
            self.chunk_narrative(narrative_text, current_section)
        
        # ===== PROCESS TABLES =====
        print("\n📊 Processing tables...")
        for idx, table in enumerate(tables):
            rows = table.get("rows", [])
            self.chunk_table(rows, f"Table {idx+1}")
            print(f"   ➜ Table {idx+1} ({len(rows)} rows)")
        
        # ===== PROCESS Q&A =====
        print("\n❓ Extracting questions and answers...")
        self.extract_qa_from_paragraphs(paragraphs)
        
        print(f"\n✅ Created {len(self.chunks)} chunks!")
    
    def extract_qa_from_paragraphs(self, paragraphs: List[Dict]):
        """Extract all Q&A pairs from paragraphs"""
        qa_pairs = []
        
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i]
            text = para.get("text", "").strip()
            is_heading = para.get("is_heading", False)
            
            # Check: Is this a question?
            if is_heading and text.startswith("Q") and ":" in text:
                question = text
                
                # Answer comes right after
                if i + 1 < len(paragraphs):
                    answer_text = paragraphs[i + 1].get("text", "").strip()
                    
                    qa_pairs.append({
                        "question": question,
                        "answer": answer_text
                    })
                    
                    i += 2  # Skip both Q and A
                else:
                    i += 1
            else:
                i += 1
        
        print(f"   ➜ Found {len(qa_pairs)} Q&A pairs")
        
        # Add each Q&A as chunk
        self.chunk_qa(qa_pairs)
    
    def save_chunks(self, output_path: str):
        """Save chunks to JSON"""
        output = {
            "total_chunks": len(self.chunks),
            "chunks": self.chunks,
            "statistics": {
                "narrative_chunks": len([c for c in self.chunks if c["type"] == "narrative"]),
                "table_chunks": len([c for c in self.chunks if c["type"] == "table"]),
                "qa_chunks": len([c for c in self.chunks if c["type"] == "qa"]),
                "total_words": sum(c["word_count"] for c in self.chunks)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """Print summary"""
        stats = {
            "narrative": len([c for c in self.chunks if c["type"] == "narrative"]),
            "table": len([c for c in self.chunks if c["type"] == "table"]),
            "qa": len([c for c in self.chunks if c["type"] == "qa"]),
        }
        
        print(f"""
╔═══════════════════════════════════════════╗
║         ✅ Chunking Complete!              ║
╚═══════════════════════════════════════════╝

📊 Results:
   • Narrative Chunks: {stats['narrative']}
   • Table Chunks: {stats['table']}
   • Q&A Chunks: {stats['qa']}
   • Total Chunks: {len(self.chunks)}

📈 Statistics:
   • Total Words: {sum(c['word_count'] for c in self.chunks)}
   • Average Chunk Size: {sum(c['word_count'] for c in self.chunks) // len(self.chunks) if self.chunks else 0} words
        """)

def main():
    extracted_json = "./data/extracted_structure.json"
    output_chunks = "./data/chunks.json"
    
    print("🚀 Starting chunk extraction...\n")
    
    chunker = SimpleChunker()
    chunker.process_document(extracted_json)
    chunker.save_chunks(output_chunks)
    chunker.print_summary()
    
    print(f"\n💾 Saved to: {output_chunks}")

if __name__ == "__main__":
    main()
