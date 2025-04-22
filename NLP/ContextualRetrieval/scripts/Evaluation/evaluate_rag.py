import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score, recall_score
from nltk.translate.bleu_score import sentence_bleu


def retrieval_eval(retrieved_chunks, relevant_chunks, k=10):
    """
    Evaluate retrieval performance using Recall@K, MRR, and NDCG.
    """
    retrieved_chunks = retrieved_chunks[:k]
    relevant_chunks_set = set(relevant_chunks)

    # Recall@K
    recall_at_k = len(set(retrieved_chunks) & relevant_chunks_set) / len(relevant_chunks_set) if len(relevant_chunks_set) > 0 else 0

    # MRR
    mrr = 0
    for i, chunk in enumerate(retrieved_chunks):
        if chunk in relevant_chunks_set:
            mrr = 1 / (i + 1) 
            break

    # NDCG
    relevance_scores = [1 if chunk in relevant_chunks_set else 0 for chunk in retrieved_chunks]
    ndcg = ndcg_score([relevance_scores], [relevance_scores]) if relevance_scores else 0

    return recall_at_k, mrr, ndcg


def augmentation_eval(generated_answer, ground_truth_answer):
    """
    Evaluate augmentation performance using BLEU score.  Handles potential errors gracefully.
    """
    try:
        bleu_score = sentence_bleu([ground_truth_answer.split()], generated_answer.split())
    except TypeError:  # Handle cases where split() fails (e.g., NoneType)
        bleu_score = 0
    except ValueError: # Handle cases where empty answers are compared
        bleu_score = 0
    return bleu_score



def evaluate_rag_system(qna_data, retrieved_chunks_list, generated_answers):
    """
    Evaluates the entire RAG system using retrieval and augmentation metrics.
    """
    results = []
    for i, (query, gt_answer) in enumerate(zip(qna_data['Question'], qna_data['Answer'])):
        retrieved_chunks = retrieved_chunks_list[i]
        generated_answer = generated_answers[i]
        relevant_chunks = [x.strip() for x in qna_data['Source Chunk Type'][i].split(',')]
        recall_k, mrr, ndcg = retrieval_eval(retrieved_chunks, relevant_chunks, k=10) # or adjust k as needed
        bleu = augmentation_eval(generated_answer, gt_answer)
        results.append([query, recall_k, mrr, ndcg, bleu])
    return pd.DataFrame(results, columns=['Query', 'Recall@K', 'MRR', 'NDCG', 'BLEU'])


