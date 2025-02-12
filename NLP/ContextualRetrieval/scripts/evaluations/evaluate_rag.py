import pandas as pd
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer, util
import mlflow

# System setup
sys.path.append(str(Path(__file__).parent.parent))
from retrieve import retrieve_documents

# Configuration
class Config:
    BASE_DIR = Path(__file__).parent.parent.parent
    QUESTIONS_PATH = BASE_DIR / "data" / "docugami" / "data" / "raw_questions" / "questions.csv"
    OUTPUT_DIR = BASE_DIR / "results"
    COLLECTIONS = ["TextChunk", "ContextChunk", "TextContextChunk"]
    SEARCH_METHODS = ["bm25", "vector", "hybrid"]
    ALPHA_VALUES = [0.3, 0.5, 0.7]
    TOP_K = 3

# Initialize evaluators
class Evaluators:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()
        self.semantic = SentenceTransformer('all-MiniLM-L6-v2')

# Document matching utilities
def normalize_doc_id(doc_id: str) -> str:
    return doc_id.split()[-1].strip('*')

def is_doc_match(retrieved_doc: str, expected_doc: str) -> bool:
    retrieved_company = normalize_doc_id(retrieved_doc)
    expected_company = normalize_doc_id(expected_doc)
    
    if len(expected_doc.split()) == 1:
        return retrieved_company == expected_company
    return retrieved_doc.strip('*') == expected_doc.strip('*')

# Core evaluation metrics
def calculate_retrieval_metrics(retrieved_docs: List[Dict], expected_docs: set) -> Dict[str, float]:
    if not retrieved_docs or not expected_docs:
        return {metric: 0.0 for metric in ["precision", "recall", "mrr", "ndcg"]}
    
    # Precision calculation
    relevant = sum(1 for doc in retrieved_docs 
                  if any(is_doc_match(doc["document"], exp_doc) for exp_doc in expected_docs))
    precision = relevant / len(retrieved_docs)
    
    # Recall calculation
    recall = relevant / len(expected_docs)
    
    # MRR calculation
    mrr = 0.0
    for rank, doc in enumerate(retrieved_docs, 1):
        if any(is_doc_match(doc["document"], exp_doc) for exp_doc in expected_docs):
            mrr = 1.0 / rank
            break
    
    # NDCG calculation
    dcg = sum(1.0 / np.log2(rank + 1) for rank, doc in enumerate(retrieved_docs, 1)
              if any(is_doc_match(doc["document"], exp_doc) for exp_doc in expected_docs))
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(expected_docs) + 1, 
                                                                len(retrieved_docs) + 1)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg
    }

class RAGEvaluator:
    def __init__(self):
        self.evaluators = Evaluators()
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def evaluate_query(self, query_data: Tuple[str, set, str]) -> Dict:
        query_text, expected_docs, question_type = query_data
        results = []

        for collection in Config.COLLECTIONS:
            collection_metrics = defaultdict(dict)

            for search_type in Config.SEARCH_METHODS:
                if search_type == "hybrid":
                    for alpha in Config.ALPHA_VALUES:
                        docs = retrieve_documents(query_text, collection, search_type, 
                                               alpha, Config.TOP_K)
                        # Calculate metrics for each alpha value separately
                        retrieval_metrics = calculate_retrieval_metrics(docs, expected_docs)
                        
                        # Calculate semantic similarity
                        if docs:
                            contexts = [doc.get("content", "") for doc in docs]
                            query_emb = self.evaluators.semantic.encode(query_text, convert_to_tensor=True)
                            context_embs = self.evaluators.semantic.encode(contexts, convert_to_tensor=True)
                            similarities = util.pytorch_cos_sim(query_emb, context_embs)[0]
                            semantic_metrics = {
                                "max_similarity": float(similarities.max()),
                                "mean_similarity": float(similarities.mean()),
                            }
                        else:
                            semantic_metrics = {"max_similarity": 0.0, "mean_similarity": 0.0}

                        # Store metrics with alpha value in key
                        collection_metrics[f"{search_type}_alpha_{alpha}"] = {
                            **retrieval_metrics,
                            **semantic_metrics,
                            "question_type_match": self._question_type_match(docs, question_type)
                        }
                else:
                    docs = retrieve_documents(query_text, collection, search_type, 
                                           limit=Config.TOP_K)
                    retrieval_metrics = calculate_retrieval_metrics(docs, expected_docs)
                    
                    if docs:
                        contexts = [doc.get("content", "") for doc in docs]
                        query_emb = self.evaluators.semantic.encode(query_text, convert_to_tensor=True)
                        context_embs = self.evaluators.semantic.encode(contexts, convert_to_tensor=True)
                        similarities = util.pytorch_cos_sim(query_emb, context_embs)[0]
                        semantic_metrics = {
                            "max_similarity": float(similarities.max()),
                            "mean_similarity": float(similarities.mean()),
                        }
                    else:
                        semantic_metrics = {"max_similarity": 0.0, "mean_similarity": 0.0}

                    collection_metrics[search_type] = {
                        **retrieval_metrics,
                        **semantic_metrics,
                        "question_type_match": self._question_type_match(docs, question_type)
                    }

            results.append({
                "query": query_text,
                "collection": collection,
                "metrics": dict(collection_metrics)
            })

        return results

    # Rest of the class remains the same
    def _question_type_match(self, retrieved_docs: List[Dict], expected_type: str) -> float:
        doc_chunks = defaultdict(set)
        for doc in retrieved_docs:
            doc_chunks[doc["document"]].add(doc["chunk_id"])

        if expected_type == "Single-Doc Single-Chunk RAG":
            return 1.0 if len(doc_chunks) == 1 and all(len(chunks) == 1 
                                                      for chunks in doc_chunks.values()) else 0.0
        elif expected_type == "Single-Doc Multi-Chunk RAG":
            return 1.0 if len(doc_chunks) == 1 and any(len(chunks) > 1 
                                                      for chunks in doc_chunks.values()) else 0.0
        elif expected_type == "Multi-Doc RAG":
            return 1.0 if len(doc_chunks) > 1 else 0.0
        return 0.0

    def run_evaluation(self) -> pd.DataFrame:
        questions_df = pd.read_csv(Config.QUESTIONS_PATH)
        
        query_data = [
            (row["Question"], 
             set(row["Source Docs"].strip('*').split(",")), 
             row["Question Type"])
            for _, row in questions_df.iterrows()
        ]

        all_results = []
        for query in tqdm(query_data, desc="Evaluating Queries"):
            all_results.extend(self.evaluate_query(query))

        return self.aggregate_results(all_results)

    def aggregate_results(self, results: List[Dict]) -> pd.DataFrame:
        aggregated = []

        for result in results:
            query = result["query"]
            collection = result["collection"]
            metrics = result["metrics"]

            for search_type, metric_values in metrics.items():
                row = {
                    "Query": query,
                    "Collection": collection,
                    "Search Type": search_type
                }
                row.update({f"Avg {k}": v for k, v in metric_values.items()})
                aggregated.append(row)

        return pd.DataFrame(aggregated)

if __name__ == "__main__":
    print("Starting Enhanced RAG Evaluation with MLflow...")
    
    mlflow.set_experiment("RAG_Evaluation")
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("top_k", Config.TOP_K)
        mlflow.log_param("collections", Config.COLLECTIONS)
        mlflow.log_param("search_methods", Config.SEARCH_METHODS)
        mlflow.log_param("alpha_values", Config.ALPHA_VALUES)

        # Run evaluation
        evaluator = RAGEvaluator()
        final_df = evaluator.run_evaluation()

        # Log metrics
        metrics = final_df.groupby("Search Type").mean().to_dict()
        for search_type, metric_values in metrics.items():
            for metric_name, metric_value in metric_values.items():
                mlflow.log_metric(f"{search_type}_{metric_name}", metric_value)

        # Save and log artifacts
        output_file = Config.OUTPUT_DIR / "mlenhanced_evaluation_metrics.csv"
        final_df.to_csv(output_file, index=False)
        mlflow.log_artifact(str(output_file))

        print(f"MLflow run completed with run_id {run.info.run_id}")