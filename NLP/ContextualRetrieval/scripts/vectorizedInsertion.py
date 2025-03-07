import os
import json
import time
import uuid
from pathlib import Path
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
WEAVIATE_CLOUD_URL = os.getenv("WEAVIATE_CLOUD_URL2")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR", "data/processed")
STANDARD_FILE = f"{DATA_DIR}/standard_rag.json"
CONTEXTUAL_FILE = f"{DATA_DIR}/contextual_rag.json"
STANDARD_ITEMS_FILE = f"{DATA_DIR}/standard_items.json"
CONTEXTUAL_ITEMS_FILE = f"{DATA_DIR}/contextual_items.json"

# Initialize OpenAI client
client_openai = OpenAI(api_key=OPENAI_API_KEY)
def get_embedding(text):
    """Return an embedding vector for given text using OpenAI's API."""
    try:
        response = client_openai.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

def generate_uuid5(data_row):
    """Generate a UUID5 from the data row using a namespace."""
    namespace = uuid.NAMESPACE_DNS
    unique_string = str(data_row.get('chunk_id', '')) + str(data_row.get('source_doc', ''))
    return str(uuid.uuid5(namespace, unique_string))

def batch_insert(collection, data, client,batch_size=500, max_retries=3):
    """Insert data into Weaviate with retries and reduced batch size."""

    collection = client.collections.get(collection)

    with collection.batch.dynamic() as batch:
        for data_row in data:
            obj_uuid = generate_uuid5(data_row)
            batch.add_object(
                properties=data_row,
                uuid=obj_uuid
            )
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    failed_objects = collection.batch.failed_objects
    if failed_objects:
        print(f"Number of failed imports: {len(failed_objects)}")
        print(f"First failed object: {failed_objects[0]}")

def load_json(file_path):
    """Loads JSON data if the file exists."""
    if Path(file_path).exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def vectorize_chunks(file_path, vectorize_function):
    """Loads and vectorizes chunks in parallel."""
    existing_data = load_json(file_path.replace("rag.json", "items.json"))
    if existing_data:
        print(f"Using precomputed embeddings from {file_path.replace('rag.json', 'items.json')}.")
        return existing_data
    
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"Vectorizing {len(chunks)} chunks...")
    vectorized_chunks = []
    
    with ThreadPoolExecutor() as executor:
        future_to_chunk = {executor.submit(vectorize_function, chunk): chunk for chunk in chunks}
        for future in as_completed(future_to_chunk):
            vectorized_chunks.append(future.result())
    
    with open(file_path.replace("rag.json", "items.json"), "w", encoding="utf-8") as f:
        json.dump(vectorized_chunks, f, indent=4, ensure_ascii=False)
    
    return vectorized_chunks

def process_standard_chunk(chunk):
    text = chunk.get("content", "").strip()
    chunk["embedding"] = get_embedding(text) if text else []
    chunk["source_type"] = "text"
    return chunk

def process_contextual_chunk(chunk):
    combined_text = f"{chunk.get('context', '')} {chunk.get('content', '')}".strip()
    chunk["embedding"] = get_embedding(combined_text) if combined_text else []
    chunk.pop("tables_context", None)  # Remove large fields if present
    return chunk

def main():
    # Connect to Weaviate
    try:
        client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_CLOUD_URL, auth_credentials=Auth.api_key(WEAVIATE_API_KEY),  skip_init_checks=True,)
        if not client.is_ready():
            print("Weaviate connection failed.")
            return
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        return
    print("Connected to Weaviate.")

    # Ensure collections exist
    for collection_name, properties in [
        ("StandardChunks", [
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="source_doc", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
            Property(name="content", data_type=DataType.TEXT),
            Property(name="embedding", data_type=DataType.NUMBER_ARRAY)
        ]),
        ("ContextualChunks", [
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="source_doc", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
            Property(name="content", data_type=DataType.TEXT),
            Property(name="context", data_type=DataType.TEXT),
            Property(name="embedding", data_type=DataType.NUMBER_ARRAY)
        ])
    ]:
        try:
            client.collections.create(collection_name, vectorizer_config=Configure.Vectorizer.none(), properties=properties)
        except:
            print(f"Collection '{collection_name}' already exists.")
    
    # Vectorize data only if embeddings are not precomputed
    try:
        standard_chunks = vectorize_chunks(STANDARD_FILE, process_standard_chunk)
        contextual_chunks = vectorize_chunks(CONTEXTUAL_FILE, process_contextual_chunk)

        # Insert into Weaviate
        batch_insert("StandardChunks", standard_chunks, client)
        batch_insert("ContextualChunks", contextual_chunks, client)

        print("Data insertion complete.")
    except Exception as e:
        print(f"Error processing or inserting data: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()

