import fitz  # PyMuPDF for text extraction
import pdfplumber  # For table extraction
import re
import json
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path

# Download NLTK tokenizer
nltk.download("punkt")

# Directories
PDF_DIR = Path("/Users/omprakashgunja/Documents/GitHub/AI-Experiments-2025/AI-Experiments-2025/NLP/Window_vs_Retrieval/data/docugami/data/v1/docs")  # Directory for input PDFs
OUTPUT_DIR = Path("processed")  # Output directory for structured data
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
CHUNK_SIZE = 5  # Number of sentences per chunk
TABLE_EXTRACTION_METHOD = "pdfplumber"  # Options: 'pdfplumber', 'camelot'


def clean_text(text):
    """Cleans extracted text by normalizing spaces and removing unwanted artifacts."""
    text = re.sub(r'\n+', '\n', text)  # Normalize multiple newlines
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'Page\s*\d+', '', text)  # Remove page numbers
    return text.strip()


def extract_text_from_pdf(pdf_path):
    """Extracts text while preserving document structure (titles, sections)."""
    doc = fitz.open(pdf_path)
    text_data = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        cleaned_text = clean_text(text)
        text_data.append({"page": page_num + 1, "text": cleaned_text})
    
    return text_data


def extract_tables_from_pdf(pdf_path):
    """Extracts tables from PDFs and stores them separately."""
    tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            table_data = page.extract_table()
            if table_data:
                tables.append({
                    "chunk_id": f"table_{page_num+1}",
                    "page": page_num + 1,
                    "source_type": "table",
                    "table_data": table_data,
                    "related_text_chunks": []  # To be linked later
                })
    
    return tables


def chunk_text(text_data, tables, doc_name):
    """Chunks text into meaningful segments based on sentences and links to tables."""
    chunks = []
    metadata = []

    for page_data in text_data:
        sentences = sent_tokenize(page_data["text"])
        related_tables = [t["chunk_id"] for t in tables if t["page"] == page_data["page"]]

        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk_text = " ".join(sentences[i:i + CHUNK_SIZE])
            chunk_id = f"{doc_name}_p{page_data['page']:02d}_c{i // CHUNK_SIZE:03d}"

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "source_doc": doc_name,
                    "page": page_data["page"],
                    "source_type": "text",
                    "content": chunk_text,
                    "related_tables": related_tables  # Linking to tables
                })

                # Update tables to reference the text chunk
                for table in tables:
                    if table["chunk_id"] in related_tables:
                        table["related_text_chunks"].append(chunk_id)
    
    return chunks


def save_json(data, filename):
    """Saves extracted data as JSON."""
    with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def process_pdfs():
    """Processes all PDFs in the input directory."""
    all_metadata = []

    for pdf_file in PDF_DIR.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")
        doc_name = pdf_file.stem
        
        # Extract text
        text_data = extract_text_from_pdf(pdf_file)
        
        # Extract tables
        table_data = extract_tables_from_pdf(pdf_file)
        
        # Chunk text and establish links to tables
        text_chunks = chunk_text(text_data, table_data, doc_name)
        
        # Combine everything into one structured JSON
        all_metadata.append({
            "source_doc": doc_name,
            "text_chunks": text_chunks,
            "tables": table_data
        })
    
    # Save final structured JSON
    save_json(all_metadata, "all_metadata.json")
    print("Processing complete.")


if __name__ == "__main__":
    process_pdfs()
