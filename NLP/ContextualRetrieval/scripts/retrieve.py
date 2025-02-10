import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
import mlflow
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
    """
    Retrieves relevant documents based on the given query and retrieval method.
    
    Parameters:
        query_text (str): The query for retrieval.
        collection_name (str): The name of the collection to search.
        search_type (str): The retrieval method ("bm25", "vector", "hybrid").
        alpha (float): The hybrid search weight (only used if search_type="hybrid").
        limit (int): Number of documents to retrieve.

    Returns:
        list: A list of retrieved document texts.
    """
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

    # Extract results
    retrieved_texts = []
    for item in response.objects:
        if collection_name == "TextChunk":
            retrieved_texts.append(item.properties.get("text", ""))
        elif collection_name == "ContextChunk":
            retrieved_texts.append(item.properties.get("contextual_description", ""))
        elif collection_name == "TextContextChunk":
            retrieved_texts.append(item.properties.get("text_context", ""))
    
    return retrieved_texts

# Example usage
if __name__ == "__main__":
    query_text = "Intel quarterly earnings report"
    
    # Retrieve from all collections and print results
    for collection in collections:
        print(f"\nRetrieving from {collection} using Hybrid Search:")
        results = retrieve_documents(query_text, collection, search_type="hybrid", alpha=0.5)
        for i, text in enumerate(results, 1):
            print(f"{i}. {text[:200]}...")  # Print first 200 chars
    
    client.close()