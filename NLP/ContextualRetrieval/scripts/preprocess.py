import json
import argparse
import openai
import os
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# Config
DATA_DIR = Path(os.getenv("PROCESSED_DIR", "data/processed"))
INPUT_FILE = DATA_DIR / "all_metadata_cleaned.json"
STANDARD_OUTPUT = DATA_DIR / "standard_rag.json"
CONTEXTUAL_OUTPUT = DATA_DIR / "contextual_rag.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
deepseekClient = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def call_openai(prompt):
    """Calls OpenAI API to generate contextual enrichment."""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in document retrieval. Generate succinct context for search retrieval."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def extract_table_data(chunk_id, tables):
    """Fetches the full related table data for a given chunk."""
    return [table for table in tables if chunk_id in table.get("related_text_chunks", [])]

def extract_table_summary(chunk_id, tables):
    """Creates a structured text representation of relevant tables.
    
    Args:
        chunk_id (str): The ID of the chunk to find related tables
        tables (list): List of table dictionaries
        
    Returns:
        str: Formatted table summary or empty string if no tables found
    """
    if not tables:
        return ""
        
    summaries = []
    for table in tables:
        if chunk_id not in table.get("related_text_chunks", []):
            continue
            
        table_data = table.get("table_data", [])
        if not table_data:
            continue
            
        # Extract headers (assuming first row contains headers)
        headers = [str(col) for col in table_data[0] if col]
        
        # Format data rows
        formatted_rows = []
        for row in table_data[1:]:  # Skip header row
            # Filter out empty cells and format each row
            row_data = [str(cell) for cell in row if cell]
            if row_data:
                formatted_rows.append(" | ".join(row_data))
        
        # Combine headers and rows
        if headers:
            summary = f"Table Headers: {' | '.join(headers)}\n"
            summary += f"Data: {'; '.join(formatted_rows)}"
            summaries.append(summary)
    
    return "Relevant tables:\n" + "\n---\n".join(summaries) if summaries else ""

def enrich_chunk(chunk, title, section, prev_chunk, next_chunk, tables):
    """Generates a succinct contextual enrichment for a given chunk using OpenAI API."""
    table_summaries = extract_table_summary(chunk["chunk_id"], tables)
    
    prompt = f"""
    <document>
    Title: {title}
    Section: {section}
    Relevant Context:
    - Previous: {prev_chunk.get('content', '') if prev_chunk else ''}
    - Next: {next_chunk.get('content', '') if next_chunk else ''}
    - Tables: {table_summaries}
    </document>
    
    <chunk>
    {chunk['content']}
    </chunk>
    

    Please provide a **short, retrieval-optimized** context that situates this chunk **within the document flow**. Ensure it:
    1. **Clearly links to Previous & Next chunks** (where relevant).
    2. **Explicitly integrates table data** (if available).
    3. **Optimizes for search retrieval**—phrasing should make it easier to find.
    4. **Avoids unnecessary repetition** of raw chunk content.

    Example: 
    Original chunk: "The company's revenue grew by 3% over the previous quarter."
    Contextualized chunk: "In Q2 2023, ACME Corp’s revenue increased by 3% from $314M in Q1. Previous data indicates steady growth in customer demand."

    Answer only with the succinct context and nothing else.
    """
    if table_summaries:
        print(f"Chunk ID: {chunk['chunk_id']}\n{table_summaries}")
    return call_openai(prompt)

def process_chunks(data):
    """Processes chunks to generate both Standard and Contextual RAG datasets."""
    standard_chunks = []
    contextual_chunks = []
    
    print("Processing chunks with LLM enrichment...")
    for doc_idx, doc in enumerate(data):
        title = doc.get("source_doc", "")
        text_chunks = doc["text_chunks"]
        tables = doc.get("tables", [])
        
        for i, chunk in enumerate(tqdm(text_chunks, desc=f"Doc {doc_idx}: Processing chunks", unit="chunk", leave=False)):
            
            chunk_copy = chunk.copy()
            chunk_copy["related_tables"] = extract_table_data(chunk["chunk_id"], tables)
            standard_chunks.append(chunk_copy)  # Standard RAG (structured raw chunks)
            
            # Add context from neighboring chunks
            prev_chunk = text_chunks[i-1] if i > 0 else None
            next_chunk = text_chunks[i+1] if i < len(text_chunks) - 1 else None
            enriched_context = enrich_chunk(chunk, title, chunk.get("section", ""), prev_chunk, next_chunk, tables)
            
            enriched_chunk = chunk.copy()
            enriched_chunk["context"] = enriched_context
            enriched_chunk["tables_context"] = extract_table_summary(chunk["chunk_id"], tables)
            enriched_chunk["related_tables"] = extract_table_data(chunk["chunk_id"], tables)
            
            contextual_chunks.append(enriched_chunk)
            # break  # For testing, remove to process all chunks
    
    return standard_chunks, contextual_chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="Input JSON file")
    parser.add_argument("--standard_output", type=str, default=STANDARD_OUTPUT, help="Standard RAG output JSON")
    parser.add_argument("--contextual_output", type=str, default=CONTEXTUAL_OUTPUT, help="Contextual RAG output JSON")
    args = parser.parse_args()
    
    print("Loading data...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    standard_chunks, contextual_chunks = process_chunks(data)
    
    print("Saving Standard RAG chunks...")
    with open(args.standard_output, "w", encoding="utf-8") as f:
        json.dump(standard_chunks, f, indent=4, ensure_ascii=False)
    
    print("Saving Contextual RAG chunks...")
    with open(args.contextual_output, "w", encoding="utf-8") as f:
        json.dump(contextual_chunks, f, indent=4, ensure_ascii=False)
    
    print("Preprocessing with LLM-based enrichment complete!")

if __name__ == "__main__":
    main()