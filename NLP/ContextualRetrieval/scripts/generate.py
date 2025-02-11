from retrieve import retrieve_documents


# Define query
query_text = "Amazon quarterly earnings report"

# Retrieve documents using BM25
retrieved_texts_bm25 = retrieve_documents(query_text, collection_name='' , search_type="bm25")
print("BM25 Retrieval:")
for i, text in enumerate(retrieved_texts_bm25, 1):
    print(f"Document {i}: {text}")
    
# Retrieve documents using Vector Search
retrieved_texts_vector = retrieve_documents(query_text, search_type="vector")
print("\nVector Search Retrieval:")
for i, text in enumerate(retrieved_texts_vector, 1):
    print(f"Document {i}: {text}")
    
# Retrieve documents using Hybrid Search
retrieved_texts_hybrid = retrieve_documents(query_text, search_type="hybrid", alpha=0.5)
print("\nHybrid Search Retrieval:")
for i, text in enumerate(retrieved_texts_hybrid, 1):
    print(f"Document {i}: {text}")
    
# Retrieve documents using Hybrid Search with different alpha value
retrieved_texts_hybrid_alpha = retrieve_documents(query_text, search_type="hybrid", alpha=0.8)
print("\nHybrid Search Retrieval (Alpha=0.8):")
for i, text in enumerate(retrieved_texts_hybrid_alpha, 1):
    print(f"Document {i}: {text}")
    
# Retrieve documents using Hybrid Search with different alpha value
retrieved_texts_hybrid_alpha = retrieve_documents(query_text, search_type="hybrid", alpha=0.2)
print("\nHybrid Search Retrieval (Alpha=0.2):")
for i, text in enumerate(retrieved_texts_hybrid_alpha, 1):
    print(f"Document {i}: {text}")
    
    