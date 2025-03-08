import chromadb as c

# Initialize ChromaDB client
chroma_client = c.PersistentClient(path="./chroma_db")

# List all collections
collections = chroma_client.list_collections()
print("\n🔍 Available Collections in ChromaDB:")
for col in collections:
    print(f"- {col.name}")

def inspect_collection(collection_name):
    """Retrieve a sample document to inspect the collection's format."""
    collection = chroma_client.get_or_create_collection(collection_name)

    # Fetch a few sample objects
    results = collection.query(query_texts=["test"], n_results=1)

    if results and results["metadatas"] and results["documents"]:
        print(f"\n📌 Collection: {collection_name}")
        print("📝 Sample Document Content:", results["documents"][0][0])
        print("📊 Metadata Fields:", results["metadatas"][0][0].keys())
    else:
        print(f"\n⚠️ No sample data found in {collection_name}")

# Inspect both collections
inspect_collection("StandardChunks")
inspect_collection("ContextualChunks")