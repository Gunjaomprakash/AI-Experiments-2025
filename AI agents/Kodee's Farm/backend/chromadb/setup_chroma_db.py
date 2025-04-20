# File: database/setup_chroma_db.py

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
from google.api_core import retry
import json
import os
# Import the tool usage recorder
from ..tools.tools import _record_tool_usage

# Retry wrapper for Gemini API calls
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
class GeminiEmbeddingFunction(EmbeddingFunction):
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=embedding_task),
        )
        return [e.values for e in response.embeddings]

# Initialize embedding function
embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

# Initialize Chroma client and collection
chroma_client = chromadb.PersistentClient(path="./backend/chromadb")
DB_NAME = "kodee_farm_db"
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

# Define documents
with open("backend/chromadb/data.json", "r") as f:
    _documents = json.load(f)["documents"]

_metadatas = [
    {"field": "Field 1", "year": "2023", "crop": "Soybeans"},
    {"field": "Field 2", "year": "2023", "crop": "Corn"},
    {"field": "Field 3", "year": "2023", "crop": "Potatoes"},
    {"field": "Field 1", "year": "2024", "crop": "Corn"},
    {"field": "Field 2", "year": "2024", "crop": "Potatoes"},
    {"field": "Field 3", "year": "2024", "crop": "Paddy"}
]

_ids = ["field1_2023", "field2_2023", "field3_2023", "field1_2024", "field2_2024", "field3_2024"]

# Add documents to collection if empty
if db.count() == 0:
    db.add(documents=_documents, ids=_ids, metadatas=_metadatas)

# Switch to query mode after setup
embed_fn.document_mode = False

# Exported utility functions
def query_kodee_memory(user_query: str):
    result = db.query(query_texts=[user_query], n_results=3)
    [retrieved_passages] = result["documents"]

    query_oneline = user_query.replace("\n", " ")
    prompt = f"""You are an intelligent farming assistant helping with agriculture insights.
Use only the provided passages to answer the user's question.

QUESTION: {query_oneline}
"""
    for passage in retrieved_passages:
        passage_clean = passage.replace('\n', ' ')
        prompt += f"PASSAGE: {passage_clean}\n"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return response.text

def queryKodeeMemories(user_query: str):
    result = db.query(query_texts=[user_query], n_results=3)
    _record_tool_usage("queryKodeeMemories")
    return result["documents"]


# Expose db object too
__all__ = ["db", "query_kodee_memory", "queryKodeeMemories"]
