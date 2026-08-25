from src.processing.chunker import PharmaChunker
import json
import os
import glob

def test_chunking():
    chunker = PharmaChunker()
    
    # metrics
    files = glob.glob("data/raw/*.json")
    if not files:
        print("No files found in data/raw/")
        return

    # Pick a complex one if available, else first
    target_file = next((f for f in files if "Mekinist" in f), files[0])
    
    print(f"Testing chunker on: {target_file}")
    
    with open(target_file, "r") as f:
        data = json.load(f)
        
    chunks = chunker.chunk_drug_label(data)
    
    print(f"\nGenerated {len(chunks)} chunks.")
    
    if chunks:
        print("\n--- SAMPLE CHUNK 0 ---")
        print(f"METADATA: {chunks[0].metadata}")
        print("-" * 20)
        print(chunks[0].page_content[:1000] + "...") # Preview
        print("-" * 20)

if __name__ == "__main__":
    test_chunking()
