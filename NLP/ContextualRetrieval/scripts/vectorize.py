
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
import os
import json
from pathlib import Path

WEAVIATE_CLOUD_URL = os.getenv("WEAVIATE_CLOUD_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

METADATA_FILE = Path("NLP/ContextualRetrieval/data/processed/metadata.json")
METADATA_CONTEXT_FILE = Path("NLP/ContextualRetrieval/data/processed/metadata_contextual.json")

def retrieve_with_context(chunk):
    if not isinstance(chunk, dict):
        return "Error: Invalid chunk format"
        
    text = chunk.get("text", "")
    context = chunk.get("contextual_description", "")
    
    if not text and not context:
        return "No text or context available"
        
    return f"Text:{text.strip()} \n Context:{context.strip()}".strip()

def batch_insert(collection, data, batch_size=100):
    """Inserts data into a Weaviate collection in smaller batches."""
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        try:
            collection.data.insert_many(batch)
            print(f"Inserted {len(batch)} items successfully.")
        except Exception as e:
            print(f"Batch insert failed: {e}")


# Connect to Weaviate Cloud
with weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLOUD_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    headers={'X-OpenAI-Api-key': OPENAI_API_KEY}
) as client:

    if client.is_ready():
        print("Weaviate connection successful.")

    # Define TextChunk schema using v4 syntax
    if not client.collections.exists("TextChunk"):
        client.collections.create(
            name="TextChunk",
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_openai(),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="document", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="text", data_type=DataType.TEXT),
            ]
        )
        print("TextChunk schema created.")

    # Define ContextChunk schema using v4 syntax
    if not client.collections.exists("ContextChunk"):
        client.collections.create(
            name="ContextChunk",
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_openai(),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="document", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="contextual_description", data_type=DataType.TEXT),
            ]
        )
        print("ContextChunk schema created.")

    if not client.collections.exists("TextContextChunk"):
        client.collections.create(
                    name="TextContextChunk",
                    vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_openai(),
                    properties=[
                        Property(name="chunk_id", data_type=DataType.TEXT),
                        Property(name="document", data_type=DataType.TEXT),
                        Property(name="page", data_type=DataType.INT),
                        Property(name="text_context", data_type=DataType.TEXT),
                    ]
                )
        print("TextContextChunk schema created.")
    # Load metadata JSON files
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    with open(METADATA_CONTEXT_FILE, "r") as f:
        contextual_metadata = json.load(f)

    # Insert `TextChunk` data
    text_chunk_collection = client.collections.get("TextChunk")
    batch_insert(text_chunk_collection, [
        {
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "page": chunk["page"],
            "text": chunk["text"],
        }
        for chunk in metadata
    ])
    print("Inserted data into TextChunk.")

    # Insert `ContextChunk` data
    context_chunk_collection = client.collections.get("ContextChunk")
    batch_insert(context_chunk_collection, [
        {
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "page": chunk["page"],
            "contextual_description": chunk["contextual_description"],
        }
        for chunk in contextual_metadata
    ])
    print("Inserted data into ContextChunk.")
    
    #insert TextContextChunk data
    text_context_chunk_collection = client.collections.get("TextContextChunk")
    batch_insert(text_context_chunk_collection, [
        {
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "page": chunk["page"],
            "text_context": retrieve_with_context(chunk)
        }
        for chunk in contextual_metadata
    ])

    print("Schema creation and data insertion complete.")