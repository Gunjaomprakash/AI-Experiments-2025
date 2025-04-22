import chromadb
import numpy as np
from openai import OpenAI
import os

# Load environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="../chroma_db")

# Load the ChromaDB collections
standard_collection = chroma_client.get_or_create_collection("StandardChunks")
contextual_collection = chroma_client.get_or_create_collection("ContextualChunks")

# Initialize OpenAI client
# client_openai = OpenAI(api_key=OPENAI_API_KEY)
def get_embedding(text, llm):
    """Generate an embedding using LLM api"""
    response = llm.client.embeddings.create(input=text, model=llm.model)
    return response.data[0].embedding

def retrieve_chunks(query_text, llm, top_k=3):
    """Retrieve relevant chunks from ChromaDB along with full metadata."""
    query_embedding = get_embedding(query_text, llm)
    
    print("\nDebugging Chunks Retrieval...")
    print(f"Number of documents in StandardChunks collection: {standard_collection.count()}")

    print(f"Number of documents in ContextualChunks collection: {contextual_collection.count()}")
    
    results_standard = standard_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    results_contextual = contextual_collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved_chunks = {
        "standard": [],
        "contextual": []
    }

    # Extract results from StandardChunks
    if results_standard and results_standard["documents"]:
        for doc, metadata in zip(results_standard["documents"][0], results_standard["metadatas"][0]):
            chunk_info = {
                "chunk_id": metadata["chunk_id"],
                "source_doc": metadata["source_doc"],
                "page": metadata["page"],
                "content": doc,
                "related_tables": metadata.get("related_tables", "N/A"),
                "collection": "StandardChunks"
            }
            retrieved_chunks["standard"].append(chunk_info)

    # Extract results from ContextualChunks
    if results_contextual and results_contextual["documents"]:
        for doc, metadata in zip(results_contextual["documents"][0], results_contextual["metadatas"][0]):
            chunk_info = {
                "chunk_id": metadata["chunk_id"],
                "source_doc": metadata["source_doc"],
                "page": metadata["page"],
                "content": doc,
                "related_tables": metadata.get("related_tables", "N/A"),
                "collection": "ContextualChunks"
            }
            retrieved_chunks["contextual"].append(chunk_info)

    return retrieved_chunks

# If running directly
if __name__ == "__main__":
    query = input("Enter your query: ")
    chunks = retrieve_chunks(query, top_k=10)