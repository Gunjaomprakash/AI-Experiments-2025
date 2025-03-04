import json
import re

def chunk_document(metadata_file, chunk_size=1000, overlap=100):
    """
    Chunks documents based on metadata, aiming for better context retention.

    Args:
        metadata_file (str): Path to the metadata.json file.
        chunk_size (int): Desired size of each chunk in characters.
        overlap (int): Number of overlapping characters between chunks.

    Returns:
        list: A list of dictionaries, each representing a chunk with relevant metadata.
    """

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    all_chunks = []
    for doc_info in metadata:
        document_text = doc_info["text"]
        document_id = doc_info["document"]
        page_number = doc_info["page"]

        # split logic
        chunks = split_text_with_overlap(document_text, chunk_size, overlap)
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}_p{page_number}_c{i:03d}"
            new_chunk_info = {
                "chunk_id": chunk_id,
                "document": document_id,
                "page": page_number,
                "text": chunk_text,
                "summary": ""  # You can add summary generation here if needed
            }
            all_chunks.append(new_chunk_info)
            
    return all_chunks


def split_text_with_overlap(text, chunk_size, overlap):
    """
    Splits a text into chunks with a specified overlap.

    Args:
        text (str): The text to split.
        chunk_size (int): The desired size of each chunk.
        overlap (int): The number of overlapping characters.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def analyze_chunks_for_question_types(chunks, metadata):
    """
    Analyzes a list of chunks and metadata to predict potential question types and sources.

    Args:
        chunks (list): A list of chunk dictionaries.
        metadata (list): raw list of metadata

    Returns:
        list: A list of dictionaries, each representing a potential question type scenario.
    """

    question_scenarios = []
    unique_documents = set()
    
    document_to_chunks = {}
    for chunk in chunks:
        doc_name = chunk['document']
        unique_documents.add(doc_name)
        if doc_name not in document_to_chunks:
            document_to_chunks[doc_name] = []
        document_to_chunks[doc_name].append(chunk)


    # Multi-document scenarios
    if len(unique_documents) > 1:
        question_scenarios.append({
            "source_docs": list(unique_documents),
            "question_type": "Multi-doc",
            "source_type": "text", # default for now
            "comment": "Question might require information from multiple documents"
        })
        
    # single doc multi chunks 
    for doc_name, doc_chunks in document_to_chunks.items():
        if len(doc_chunks) > 1:
            question_scenarios.append({
                "source_docs": [doc_name],
                "question_type": "Single Doc Multi chunk",
                "source_type": "text", # default for now
                "comment": f"Question might require information from multiple chunks within {doc_name}."
            })
        
        # Single-document, single-chunk scenarios
        if len(doc_chunks) >= 1:
            question_scenarios.append({
                "source_docs": [doc_name],
                "question_type": "Single Doc Single Chunk",
                "source_type": "text",
                "comment": "Question likely contained in a single chunk"
                
            })
        
    # analyze source types
    for chunk in chunks:
        if "Table of Contents" in chunk["text"] or "Condensed Statements" in chunk["text"]:
             for scenario in question_scenarios:
                if chunk["document"] in scenario["source_docs"]:
                    scenario["source_type"] = "table"
        
    return question_scenarios


# Example Usage (replace with your actual metadata file)
metadata_file = "/Users/omprakashgunja/Documents/GitHub/AI-Experiments-2025/AI-Experiments-2025/NLP/ContextualRetrieval/data/processed/metadata.json"  
new_chunks = chunk_document(metadata_file, chunk_size=1200, overlap=150)

# Print the first few chunks to see the result
print("Sample Chunks:")
for i in range(min(5, len(new_chunks))):
    print(f"\nChunk {i+1}:")
    print(f"  Chunk ID: {new_chunks[i]['chunk_id']}")
    print(f"  Document: {new_chunks[i]['document']}")
    print(f"  Page: {new_chunks[i]['page']}")
    print(f"  Text (first 100 chars): {new_chunks[i]['text'][:100]}...")

# Analyze the chunks for potential question types
print("\n\nAnalyzing for question types:")
question_type_results = analyze_chunks_for_question_types(new_chunks,metadata_file)

for scenario in question_type_results:
    print(scenario)
