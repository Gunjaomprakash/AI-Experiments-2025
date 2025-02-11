import pandas as pd
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

# Ensure we can import the retrieval function
sys.path.append(str(Path(__file__).parent.parent))
from retrieve import retrieve_documents  # Import retrieval function

# Path configuration
BASE_DIR = Path(__file__).parent.parent.parent
QUESTIONS_PATH = BASE_DIR / "data" / "docugami" / "data" / "raw_questions" / "questions.csv"

# Define retrieval settings
COLLECTIONS = ["TextChunk", "ContextChunk", "TextContextChunk"]
SEARCH_METHODS = ["bm25", "vector", "hybrid"]
ALPHA_VALUES = [0.3, 0.5, 0.7]  # For hybrid search
TOP_K = 3  # Number of retrieved results to consider


# --------------------------- METRICS CALCULATION ---------------------------- #

def source_doc_match(retrieved_metadata: List[Dict], expected_docs: set) -> float:
    """Compute Source Docs Match - Partial match checking."""
    hits = 0
    for retrieved_doc in retrieved_metadata:
        retrieved_name = retrieved_doc["document"]
        if any(expected in retrieved_name for expected in expected_docs):
            hits += 1
    return hits / len(retrieved_metadata) if retrieved_metadata else 0.0


def question_type_match(retrieved_metadata: List[Dict], expected_question_type: str) -> float:
    """Compute Question Type Match."""
    doc_chunks = defaultdict(set)
    for item in retrieved_metadata:
        doc_chunks[item["document"]].add(item["chunk_id"])

    if expected_question_type == "Single-Doc Single-Chunk RAG":
        return 1.0 if len(doc_chunks) == 1 and all(len(chunks) == 1 for chunks in doc_chunks.values()) else 0.0
    elif expected_question_type == "Single-Doc Multi-Chunk RAG":
        return 1.0 if len(doc_chunks) == 1 and any(len(chunks) > 1 for chunks in doc_chunks.values()) else 0.0
    elif expected_question_type == "Multi-Doc RAG":
        return 1.0 if len(doc_chunks) > 1 else 0.0
    return 0.0


def first_match_metric(retrieved_metadata: List[Dict], expected_docs: set) -> float:
    """Compute First Match Metric (Position of first correct document)."""
    for i, retrieved_doc in enumerate(retrieved_metadata, start=1):
        if any(expected in retrieved_doc["document"] for expected in expected_docs):
            return i
    return len(retrieved_metadata) + 1  # If no match found, return worst case


def normalized_mcc(retrieved_metadata: List[Dict], expected_docs: set) -> float:
    """Compute Normalized MCC (Matthews Correlation Coefficient)."""
    y_true = [1 if any(expected in doc["document"] for expected in expected_docs) else 0 for doc in retrieved_metadata]
    y_pred = [1] * len(retrieved_metadata)  # Assuming all retrieved docs are positive predictions

    # Ensure MCC doesn't break with single-class inputs
    if len(set(y_true)) == 1:  
        return 0.0  # MCC is undefined for single-class cases, so return neutral value

    return matthews_corrcoef(y_true, y_pred)  # Explicitly define labels


# --------------------------- EVALUATION PROCESS ---------------------------- #

def evaluate_query(query_data: Tuple[str, set, str]) -> Dict:
    """Evaluate retrieval methods for a single query."""
    query_text, expected_docs, question_type = query_data
    results = []

    for collection in COLLECTIONS:
        collection_metrics = {search: {"source_match": [], "question_type_match": [], "first_match": [], "nmcc": []}
                              for search in SEARCH_METHODS}

        for search_type in SEARCH_METHODS:
            retrieved_metadata = []
            if search_type == "hybrid":
                for alpha in ALPHA_VALUES:
                    retrieved = retrieve_documents(query_text, collection, search_type, alpha, TOP_K)
                    retrieved_metadata.extend(retrieved)
            else:
                retrieved = retrieve_documents(query_text, collection, search_type, limit=TOP_K)
                retrieved_metadata.extend(retrieved)

            # Calculate metrics
            source_match = source_doc_match(retrieved_metadata, expected_docs)
            question_match = question_type_match(retrieved_metadata, question_type)
            first_match = first_match_metric(retrieved_metadata, expected_docs)
            nmcc_score = normalized_mcc(retrieved_metadata, expected_docs)

            # Store metrics
            collection_metrics[search_type]["source_match"].append(source_match)
            collection_metrics[search_type]["question_type_match"].append(question_match)
            collection_metrics[search_type]["first_match"].append(first_match)
            collection_metrics[search_type]["nmcc"].append(nmcc_score)

        results.append({
            "query": query_text,
            "collection": collection,
            "metrics": collection_metrics
        })

    return results


def run_evaluation():
    """Run evaluation for all queries in the dataset."""
    questions_df = pd.read_csv(QUESTIONS_PATH)
    
    query_data = [
        (row["Question"], set(row["Source Docs"].strip('*').split(",")), row["Question Type"])
        for _, row in questions_df.iterrows()
    ]

    all_results = []
    for query in tqdm(query_data, desc="Evaluating Queries"):
        all_results.extend(evaluate_query(query))

    return all_results


# --------------------------- RESULTS AGGREGATION ---------------------------- #

def aggregate_results(results: List[Dict]) -> pd.DataFrame:
    """Compute final scores by averaging metrics for each method and collection."""
    aggregated = []

    for result in results:
        query = result["query"]
        collection = result["collection"]
        metrics = result["metrics"]

        for search_type, metric_values in metrics.items():
            avg_source_match = np.mean(metric_values["source_match"])
            avg_question_match = np.mean(metric_values["question_type_match"])
            avg_first_match = np.mean(metric_values["first_match"])
            avg_nmcc = np.mean(metric_values["nmcc"])

            aggregated.append({
                "Query": query,
                "Collection": collection,
                "Search Type": search_type,
                "Avg Source Match": avg_source_match,
                "Avg Question Type Match": avg_question_match,
                "Avg First Match": avg_first_match,
                "Avg NMCC": avg_nmcc
            })

    return pd.DataFrame(aggregated)


# --------------------------- EXECUTION ---------------------------- #

if __name__ == "__main__":
    print("Starting RAG Evaluation...")
    
    evaluation_results = run_evaluation()
    final_df = aggregate_results(evaluation_results)

    output_file = BASE_DIR / "results" / "evaluation_metrics.csv"
    final_df.to_csv(output_file, index=False)

    print(f"Evaluation completed! Results saved to {output_file}")