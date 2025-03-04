import json
import argparse
import weaviate
import openai
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os

# Config
DATA_DIR = os.getenv("PROCESSED_DIR")
INPUT_FILE = DATA_DIR / "all_metadata_cleaned.json"
OUTPUT_FILE = DATA_DIR / "vectorized_chunks.json"

# Load embedding model
MODEL_NAME = "all-MiniLM-L6-v2"  # Change if needed
embedder = SentenceTransformer(MODEL_NAME)

def load_chunks(file_path):
    """Loads text chunks from JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_embeddings(chunks):
    """Generates vector embeddings for text chunks."""
    for doc in chunks:
        for chunk in doc["text_chunks"]:
            chunk["embedding"] = embedder.encode(chunk["content"]).tolist()
    return chunks

def save_chunks(chunks, file_path):
    """Saves vectorized chunks to JSON."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="Input JSON file")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output JSON file")
    args = parser.parse_args()
    
    print("Loading text chunks...")
    chunks = load_chunks(args.input)
    
    print("Generating embeddings...")
    vectorized_chunks = generate_embeddings(chunks)
    
    print("Saving vectorized chunks...")
    save_chunks(vectorized_chunks, args.output)
    
    print(f"Vectorized chunks saved to: {args.output}")

if __name__ == "__main__":
    main()
