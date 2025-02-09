from IPython import get_ipython
from IPython.display import display
import json
import mlflow
import os
from pathlib import Path
from openai import OpenAI  # Importing OpenAI SDK
from tqdm import tqdm
import concurrent.futures

# Initialize OpenAI Client Properly
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


METADATA_FILE = Path("metadata.json")
OUTPUT_FILE = Path("metadata_contextual.json")

# Load metadata.json (contains chunk IDs)
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

def retrieve_with_context(chunk, json_data, num_neighbors=6):
    """
    Retrieves a chunk along with its adjacent left and right neighboring chunks.
    Handles edge cases for first and last chunks.

    Args:
        chunk (dict): The full JSON chunk object.
        json_data (list): List of JSON chunk objects (sorted by page and chunk order).
        num_neighbors (int): Number of left and right chunks to retrieve.

    Returns:
        tuple: (main_chunk, left_chunks, right_chunks)
    """
    chunk_page = chunk["page"]

    # Sort chunks by (page, chunk order) for consistent indexing
    sorted_chunks = sorted(json_data, key=lambda c: (c["page"], c.get("chunk_order", 0)))

    # Find the index of the given chunk
    chunk_index = next((i for i, c in enumerate(sorted_chunks) if c["page"] == chunk_page and c["chunk_id"] == chunk["chunk_id"]), None)

    if chunk_index is None:
        raise ValueError(f"Chunk not found in data.")

    total_chunks = len(sorted_chunks)

    # Determine left and right chunk indices, ensuring they don't go out of bounds
    left_idx = max(chunk_index - num_neighbors, 0)
    right_idx = min(chunk_index + num_neighbors + 1, total_chunks)

    # Extract main chunk object
    main_chunk = sorted_chunks[chunk_index]

    # Extract left and right context chunk objects
    left_chunks = [c["text"] for c in sorted_chunks[left_idx:chunk_index]] if chunk_index > 0 else []
    right_chunks = [c["text"] for c in sorted_chunks[chunk_index + 1:right_idx]] if chunk_index < total_chunks - 1 else []

    return main_chunk, left_chunks, right_chunks

def generate_contextual_description(chunk):

    try:
        main_chunk, left_chunks, right_chunks = retrieve_with_context(chunk, metadata)

        prompt = f"""
        Here is the chunk we want to situate within the whole document: 

        <chunk> 
        {main_chunk["text"]} 
        </chunk> 

        To provide a richer understanding, here are its adjacent chunks:

        <left_chunks>
        {" ".join(left_chunks) if left_chunks else "No left context available."}
        </left_chunks>

        <right_chunks>
        {" ".join(right_chunks) if right_chunks else "No right context available."}
        </right_chunks>

        Please generate a **succinct contextual description** that situates this chunk within the document.  
        The goal is to **enhance search retrieval** by preserving meaning and coherence.

        **Rules:**
        - Try including key things from the left and right chunks if relevant.
        - Focus on **bridging the meaning** between this chunk and the overall document.
        - Use the **adjacent chunks** to refine the context.
        - **Do NOT summarize** the chunk itself.
        - **Answer ONLY with the succinct contextual description and nothing else.**
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a highly effective text analyzing assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        if response and response.choices:
            return response.choices[0].message.content
        else:
            print("Warning: No choices returned from API.")
            return ""
    except Exception as e:
        print(f"Error calling API: {e}")
        return ""

# Function to process a single chunk
def process_chunk(chunk):
    contextual_summary = generate_contextual_description(chunk)
    chunk["contextual_description"] = contextual_summary
    return chunk

# Track extra computation with MLflow
mlflow.start_run()

# Track total token usage
extra_tokens_used = 0
chunks_processed = 0

# Generate contextual descriptions using multiprocessing
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_chunk, metadata), total=len(metadata), desc="Processing chunks"))

# Update metadata with results
metadata = results

# Save updated metadata with contextual descriptions
with open(OUTPUT_FILE, "w") as f:
    json.dump(metadata, f, indent=4)

# Calculate and log metrics
extra_tokens_used = sum(len(chunk.get("contextual_description", "").split()) for chunk in metadata)
chunks_processed = len(metadata)

# Log computation cost
mlflow.log_param("extra_compute_task", "Contextual RAG Context Generation")
mlflow.log_metric("extra_tokens_used", extra_tokens_used)
mlflow.log_metric("chunks_processed", chunks_processed)

# End MLflow run
mlflow.end_run()

print(f"Contextual descriptions generated and saved to {OUTPUT_FILE}.")
print(f"Total extra tokens used: {extra_tokens_used}")
print(f"Total chunks processed: {chunks_processed}")

# Ensure MLflow session is properly closed
mlflow.end_run()
