import fitz  # PyMuPDF for PDF text extraction
import os
import json
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import torch

# device = "mps" if torch.backends.mps.is_available() else "cpu"
# summarizer = pipeline("summarization", model="t5-small", device=0 if device == "mps" else -1)

nltk.download('punkt')

# Directories
DOCS_DIR = Path("data/docugami/data/v1/docs/")
TEXT_DIR = Path("data/processed/text/")
CHUNKS_DIR = Path("data/processed/chunks/")
METADATA_FILE = Path("data/processed/metadata.json")

os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Chunking Parameters
CHUNK_SIZE = 15  # Larger chunk size for better context
SUMMARIZATION_THRESHOLD = 30  # Only summarize if chunk > 100 words

# Load a lightweight summarization model
summarizer = pipeline("summarization", model="t5-small")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file and keep page numbers."""
    doc = fitz.open(pdf_path)
    text_data = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            text_data.append({"page": page_num + 1, "text": text})

    return text_data

def summarize_text(text):
    """Generate a summary of the chunk."""
    if len(text.split()) < SUMMARIZATION_THRESHOLD:
        return text  # Skip summarization if chunk is too short
    summary = summarizer(text, max_length=60, min_length=25, do_sample=False)
    return summary[0]["summary_text"]

def chunk_text_smart(text_data, doc_name):
    """Split text into larger, non-overlapping chunks with metadata and summaries."""
    chunks = []
    metadata = []

    for page_data in text_data:
        sentences = sent_tokenize(page_data["text"])
        
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk_text = " ".join(sentences[i:i+CHUNK_SIZE])
            
            if chunk_text:
                chunk_id = f"{doc_name}_page{page_data['page']}_chunk{i//CHUNK_SIZE}"
                summary = summarize_text(chunk_text)

                chunks.append(chunk_text)
                metadata.append({
                    "chunk_id": chunk_id,
                    "document": doc_name,
                    "page": page_data["page"],
                    "text": chunk_text,
                    "summary": summary
                })

    return chunks, metadata

if __name__ == "__main__":
    all_metadata = []

    for pdf_file in DOCS_DIR.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")

        # Extract text
        text_data = extract_text_from_pdf(pdf_file)
        text_path = TEXT_DIR / (pdf_file.stem + ".txt")
        text_path.write_text("\n\n".join([p["text"] for p in text_data]))
        print(f"Extracted: {text_path}")

        # Chunk text with metadata and summaries
        chunks, metadata = chunk_text_smart(text_data, pdf_file.stem)
        chunk_path = CHUNKS_DIR / (pdf_file.stem + "_chunks.txt")
        chunk_path.write_text("\n\n".join(chunks))
        print(f"Chunked into {len(chunks)} segments: {chunk_path}")

        # Store metadata
        all_metadata.extend(metadata)

    # Save metadata to JSON
    with open(METADATA_FILE, "w") as f:
        json.dump(all_metadata, f, indent=4)

    print("Preprocessing complete with summaries and larger chunks.")
