import pandas as pd
import mlflow
import sys
import os
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
# Get absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PROJECT_ROOT)

from ..retrieve import retrieve_documents  # Import retrieval function
# Build absolute path to questions.csv
questions_path = os.path.join(PROJECT_ROOT, "data/docugami/data/raw_questions/questions.csv")
questions_df = pd.read_csv(questions_path)

# Define parameters
collections = ["TextChunk", "ContextChunk", "TextContextChunk"]
search_methods = ["bm25", "vector", "hybrid"]
alpha_values = [0.3, 0.5, 0.7]

# Start MLflow Tracking
mlflow.set_experiment("RAG Retrieval and Search Evaluation")
mlflow.start_run()

def jaccard_similarity(set1, set2):
    """Compute Jaccard Similarity between two sets."""
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def evaluate_retrieval():
    """Evaluates BM25, Vector, and Hybrid retrieval methods against ground truth."""
    results = []

    for _, row in questions_df.iterrows():
        query_text = row["Question"]
        expected_docs = set(row["Source Docs"].split(", "))  # Ground truth docs
        
        for collection in collections:
            retrieved_results = {}

            # Run searches using `retrieve_documents`
            for search_type in search_methods:
                if search_type == "hybrid":
                    retrieved_results[search_type] = {
                        alpha: set(retrieve_documents(query_text, collection, search_type, alpha)) for alpha in alpha_values
                    }
                else:
                    retrieved_results[search_type] = set(retrieve_documents(query_text, collection, search_type))

            # Compute Metrics
            bm25_accuracy = len(retrieved_results["bm25"] & expected_docs) / len(expected_docs)
            vector_accuracy = len(retrieved_results["vector"] & expected_docs) / len(expected_docs)
            hybrid_accuracies = {alpha: len(retrieved_results["hybrid"][alpha] & expected_docs) / len(expected_docs) for alpha in alpha_values}

            bm25_vs_vector_jaccard = jaccard_similarity(retrieved_results["bm25"], retrieved_results["vector"])
            bm25_vs_hybrid_jaccard = {alpha: jaccard_similarity(retrieved_results["bm25"], retrieved_results["hybrid"][alpha]) for alpha in alpha_values}
            
            # Log to MLflow
            mlflow.log_metric(f"{collection}_BM25_accuracy", bm25_accuracy)
            mlflow.log_metric(f"{collection}_Vector_accuracy", vector_accuracy)
            for alpha, acc in hybrid_accuracies.items():
                mlflow.log_metric(f"{collection}_Hybrid_{alpha}_accuracy", acc)

            mlflow.log_metric(f"{collection}_BM25_vs_Vector_Jaccard", bm25_vs_vector_jaccard)
            for alpha, jacc in bm25_vs_hybrid_jaccard.items():
                mlflow.log_metric(f"{collection}_BM25_vs_Hybrid_{alpha}_Jaccard", jacc)

            # Store results
            results.append({
                "Query": query_text,
                "Collection": collection,
                "BM25 Accuracy": bm25_accuracy,
                "Vector Accuracy": vector_accuracy,
                **{f"Hybrid {alpha} Accuracy": acc for alpha, acc in hybrid_accuracies.items()},
                "BM25 vs Vector Jaccard": bm25_vs_vector_jaccard,
                **{f"BM25 vs Hybrid {alpha} Jaccard": jacc for alpha, jacc in bm25_vs_hybrid_jaccard.items()}
            })
    
    return pd.DataFrame(results)

# Run Evaluation
retrieval_results = evaluate_retrieval()

# Save results as CSV
output_path = "NLP/ContextualRetrieval/results/retrieval_evaluation.csv"
retrieval_results.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# End MLflow Run
mlflow.end_run()