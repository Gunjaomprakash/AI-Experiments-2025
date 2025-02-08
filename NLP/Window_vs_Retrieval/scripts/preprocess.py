import fitz  # PyMuPDF
import re
import os
import json
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import torch

# Initialize NLP Components
nltk.download("punkt")

device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
summarizer = pipeline("summarization", model="t5-small", device=device)

# Directories
DOCS_DIR = Path("NLP/Window_vs_Retrieval/data/docugami/data/v1/docs")
TEXT_DIR = Path("NLP/Window_vs_Retrieval/data/processed/text")
CHUNKS_DIR = Path("NLP/Window_vs_Retrieval/data/processed/chunks")
METADATA_FILE = Path("NLP/Window_vs_Retrieval/data/processed/metadata.json")

TEXT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
CHUNK_SIZE = 10  # Number of sentences per chunk (smaller size for dense text)
SUMMARIZATION_THRESHOLD = 100  # Summarize if chunk > 50 words

def clean_text(text):
    """Preprocess extracted text by removing excessive spaces, newlines, and page numbers."""
    text = re.sub(r'\n+', '\n', text)  # Normalize multiple newlines
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'Page\s*\d+', '', text)  # Remove page numbers
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """Extracts text from PDF pages while handling table structures separately."""
    doc = fitz.open(pdf_path)
    text_data = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            cleaned_text = clean_text(text)
            text_data.append({"page": page_num + 1, "text": cleaned_text})

    return text_data

def summarize_text(text):
    """Summarizes a text chunk if it exceeds the threshold."""
    if len(text.split()) < SUMMARIZATION_THRESHOLD:
        return text  # Skip summarization if too short
    summary = summarizer(text, max_length=80, min_length=30, do_sample=False)
    return summary[0]["summary_text"] if summary else text

def chunk_text(text_data, doc_name):
    """Chunks text into meaningful segments based on sentences."""
    chunks = []
    metadata = []

    for page_data in text_data:
        sentences = sent_tokenize(page_data["text"])

        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk_text = " ".join(sentences[i:i+CHUNK_SIZE])
            
            if chunk_text:
                chunk_id = f"{doc_name}_p{page_data['page']:02d}_c{i//CHUNK_SIZE:03d}"
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
        text_path = TEXT_DIR / f"{pdf_file.stem}.txt"
        text_path.write_text("\n\n".join([p["text"] for p in text_data]))
        print(f"Extracted: {text_path}")

        # Chunk text with metadata and summaries
        chunks, metadata = chunk_text(text_data, pdf_file.stem)
        chunk_path = CHUNKS_DIR / f"{pdf_file.stem}_chunks.txt"
        chunk_path.write_text("\n\n".join(chunks))
        print(f"Chunked into {len(chunks)} segments: {chunk_path}")

        # Store metadata
        all_metadata.extend(metadata)

    # Save metadata to JSON
    METADATA_FILE.write_text(json.dumps(all_metadata, indent=4))

    print("Preprocessing complete with structured chunks.")
