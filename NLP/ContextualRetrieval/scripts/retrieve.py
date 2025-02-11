import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
import os
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)

# Load Weaviate Credentials
WEAVIATE_CLOUD_URL = os.getenv("WEAVIATE_CLOUD_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLOUD_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    headers={'X-OpenAI-Api-key': OPENAI_API_KEY}
)

# Define collections
collections = ["TextChunk", "ContextChunk", "TextContextChunk"]

def retrieve_documents(query_text, collection_name="TextChunk", search_type="hybrid", alpha=0.5, limit=5):
    collection = client.collections.get(collection_name)

    # Select retrieval method
    if search_type == "bm25":
        response = collection.query.bm25(
            query=query_text,
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )
    elif search_type == "vector":
        response = collection.query.near_text(
            query=query_text,
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )
    elif search_type == "hybrid":
        response = collection.query.hybrid(
            query=query_text,
            alpha=alpha,
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )
    else:
        raise ValueError("Invalid search_type. Choose from 'bm25', 'vector', or 'hybrid'.")

    # Extract results with metadata
    retrieved_docs = []
    for rank, item in enumerate(response.objects, 1):
        doc_metadata = {
            "chunk_id": item.properties.get("chunk_id", ""),
            "document": item.properties.get("document", ""),
            "page": item.properties.get("page", ""),
            "content": item.properties.get("text", "")
            if collection_name == "TextChunk"
            else item.properties.get("contextual_description", "")
            if collection_name == "ContextChunk"
            else item.properties.get("text_context", ""),
            "rank": rank,  # Rank based on order retrieved
        }
        retrieved_docs.append(doc_metadata)

    return retrieved_docs

if __name__ == "__main__":
    query_text = "Intel quarterly earnings report"
    
    # Retrieve from all collections and print results
    for collection in collections:
        print(f"\n🔍 Retrieving from {collection} using Hybrid Search:")
        results = retrieve_documents(query_text, collection, search_type="hybrid", alpha=0.5)

        for i, result in enumerate(results, 1):
            print(f"{i}.**Chunk ID:** {result['chunk_id']} | **Document:** {result['document']} |**Page:** {result['page']} |  **Distance:** {result['distance']}")
            print(f"**Text:** {result['text'][:200]}...")  # Print first 200 chars
    
    client.close()