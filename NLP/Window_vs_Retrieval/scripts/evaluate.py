import json
import tiktoken
from pathlib import Path

# File Paths
METADATA_FILE = Path("NLP/Window_vs_Retrieval/data/processed/metadata.json")
TEXTDATA = Path("NLP/Window_vs_Retrieval/data/processed/text")
CHUNKDATA = Path("NLP/Window_vs_Retrieval/data/processed/chunks")

# Load metadata.json
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

# Choose a tokenizer (e.g., OpenAI's GPT-4 tokenizer)
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 / GPT-3.5 tokenizer

def count_tokens_in_directory(directory):
    """Counts the total number of tokens in all text files within a directory."""
    total_tokens = 0
    for file in directory.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            total_tokens += len(tokenizer.encode(text))
    return total_tokens

# Function to count tokens
def count_tokens(text):
    return len(tokenizer.encode(text))

# Compute total tokens
total_tokens_chunk_text = sum(count_tokens(chunk["text"]) for chunk in metadata)
text_tokens = count_tokens_in_directory(TEXTDATA)
chunk_tokens = count_tokens_in_directory(CHUNKDATA)

# Display result
print(f"Estimated total tokens in dataset: {total_tokens_chunk_text}")
print(f"Total tokens in text data: {text_tokens}")
print(f"Total tokens in chunk data: {chunk_tokens}")
