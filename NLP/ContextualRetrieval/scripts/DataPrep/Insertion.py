import os
import json
import time
import uuid
from pathlib import Path
import random
import chromadb
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR", "data/processed")
STANDARD_FILE = f"{DATA_DIR}/standard_rag.json"
CONTEXTUAL_FILE = f"{DATA_DIR}/contextual_rag.json"

# Initialize ChromaDB (Local Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create collections
standard_collection = chroma_client.get_or_create_collection("StandardChunks")
contextual_collection = chroma_client.get_or_create_collection("ContextualChunks")

# Initialize OpenAI client
client_openai = OpenAI(api_key=OPENAI_API_KEY)

def get_batch_embeddings(texts, max_retries=3, retry_delay=1):
    """Returns embeddings for a batch of texts using OpenAI API with retries."""
    retries = 0
    while retries < max_retries:
        try:
            response = client_openai.embeddings.create(input=texts, model="text-embedding-3-small")
            return [res.embedding for res in response.data]
        except Exception as e:
            print(f"Embedding batch error: {e}, retrying {retries}/{max_retries}")
            retries += 1
            time.sleep(retry_delay * (retries + random.uniform(0, 1)))
    print(f"Failed to get embeddings after {max_retries} retries.")
    return [[] for _ in texts]  # Return empty embeddings if all retries fail

def preprocess_data(data_row):
    """Convert `related_tables` to a JSON string before inserting into ChromaDB."""
    if "related_tables" in data_row:
        data_row["related_tables"] = json.dumps(data_row["related_tables"], ensure_ascii=False)
    return data_row

def batch_insert(collection, data, collection_name):
    """Insert data into ChromaDB with preprocessing."""
    print(f"Inserting {len(data)} records into {collection_name}...")

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for data_row in data:
        data_row = preprocess_data(data_row)
        obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, data_row["chunk_id"]))  # Generate unique ID

        ids.append(obj_uuid)
        combined_text = f"{data_row.get('context', '')} {data_row['content']}".strip()
        documents.append(combined_text)
        embeddings.append(data_row["embedding"])
        metadatas.append({
            "chunk_id": data_row["chunk_id"],
            "source_doc": data_row["source_doc"],
            "page": data_row["page"],
            "related_tables": data_row["related_tables"]
        })

    # Add to ChromaDB collection
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    print(f"Successfully inserted {len(data)} records into {collection_name}.")

def load_json(file_path):
    """Loads JSON data if the file exists."""
    if Path(file_path).exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def vectorize_chunks(file_path, vectorize_function):
    """Loads and vectorizes chunks efficiently, skipping precomputed ones."""
    existing_data_path = file_path.replace("rag.json", "items.json")
    existing_data = load_json(existing_data_path) or []
    existing_ids = {item["chunk_id"] for item in existing_data}  # Track processed items

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    new_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in existing_ids]

    if not new_chunks:
        print(f"All embeddings exist in {existing_data_path}. Skipping reprocessing.")
        return existing_data

    print(f" Vectorizing {len(new_chunks)} new chunks...")

    # Process embeddings in batches
    texts = [vectorize_function(chunk) for chunk in new_chunks if chunk]
    texts = [t["content"] for t in texts if t]  # Extract content for embedding

    batch_size = 10  # Adjust for cost savings
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = get_batch_embeddings(batch_texts)
        embeddings.extend(batch_embeddings)

    # Add embeddings back into chunks
    for i, chunk in enumerate(new_chunks):
        chunk["embedding"] = embeddings[i]

    final_data = existing_data + new_chunks

    # Save updated data
    with open(existing_data_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    return final_data

def process_standard_chunk(chunk):
    """Process and generate embedding for standard chunks."""
    chunk["content"] = chunk.get("content", "").strip()
    return chunk

def process_contextual_chunk(chunk):
    """Process and generate embedding for contextual chunks."""
    chunk["content"] = f"{chunk.get('context', '')} {chunk.get('content', '')}".strip()
    chunk.pop("tables_context", None)  # Remove large fields if present
    return chunk

def main():
    print("Starting ChromaDB Data Insertion...")
    
    try:
        # Check if embeddings already exist before reprocessing
        standard_chunks = vectorize_chunks(STANDARD_FILE, process_standard_chunk)
        contextual_chunks = vectorize_chunks(CONTEXTUAL_FILE, process_contextual_chunk)

        # Insert into ChromaDB
        batch_insert(standard_collection, standard_chunks, "StandardChunks")
        batch_insert(contextual_collection, contextual_chunks, "ContextualChunks")

        print("Data insertion into ChromaDB is complete.")
    except Exception as e:
        print(f"Error processing or inserting data: {e}")

if __name__ == "__main__":
    main()