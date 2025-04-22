import sys
import os

# Add paths to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)  # Add the current directory
sys.path.append(os.path.join(script_dir, 'evaluation'))  # Add evaluation directory
sys.path.append(os.path.join(script_dir, 'modeling'))  # Add modeling directory
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))  # Add parent directory

# Then try your imports
from .evaluation.evaluate_rag import retrieval_eval, augmentation_eval
from modeling import retrieve, generate
from models.LLM import LLM

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# client= OpenAI(api_key=OPENAI_API_KEY)
deepseek_llm = LLM(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", model_name="deepseek-chat")
openai_llm = LLM(api_key=OPENAI_API_KEY,base_url="https://api.openai.com/v1" ,model_name="gpt-4-1106-preview")
openai_embed_llm = LLM(api_key=OPENAI_API_KEY,base_url="https://api.openai.com/v1" ,model_name="text-embedding-3-small")



def main_rag_pipeline(query):
    topk = 10
    print(f"\nStep 1: Retrieving Top {topk} Chunks...")
    retrieved_chunks = retrieve.retrieve_chunks(query, openai_embed_llm, top_k=topk)

    

    print("\nStep 2: Evaluating Retrieval (Recall@K, MRR, NDCG)...")
    standard_chunks = retrieved_chunks["standard"]
    contextual_chunks = retrieved_chunks["contextual"]

    # Evaluate standard chunks
    print("\n**Evaluating Standard Chunks:**")
    relevant_chunks_standard = []  # Replace with actual relevant chunks for standard
    recall_k_standard, mrr_standard, ndcg_standard = retrieval_eval(standard_chunks, relevant_chunks_standard, k=topk)
    print(f"Standard Chunks - Recall@K: {recall_k_standard}, MRR: {mrr_standard}, NDCG: {ndcg_standard}")

    # Evaluate contextual chunks
    print("\n**Evaluating Contextual Chunks:**")
    relevant_chunks_contextual = []  # Replace with actual relevant chunks for contextual
    recall_k_contextual, mrr_contextual, ndcg_contextual = retrieval_eval(contextual_chunks, relevant_chunks_contextual, k=topk)
    print(f"Contextual Chunks - Recall@K: {recall_k_contextual}, MRR: {mrr_contextual}, NDCG: {ndcg_contextual}")

    print("\nStep 3: Running Augmentation Evaluation...")
    print("\n**Retrieved Standard Chunks:**")
    for chunk in standard_chunks:
        print(chunk["source_doc"])

    print("\n**Retrieved Contextual Chunks:**")
    for chunk in contextual_chunks:
        print(chunk["source_doc"])

    print("\nStep 4.1: Generating Answer Using the Top 5 Standard Chunks...")
    response1 = generate.generate_response(query, standard_chunks, deepseek_llm)
    print(response1)

    print("\nStep 4.2: Generating Answer Using the Top 5 Contextual Chunks...")
    response2 = generate.generate_response(query, contextual_chunks, deepseek_llm)
    print(response2)

    # Evaluate augmentation for standard chunks
    print("\n**Evaluating Augmentation for Standard Chunks:**")
    ground_truth_answer_standard = "Apple's net sales grew steadily from 2019 to 2022, peaking at $394.3 billion, before a minor dip in 2023."
    bleu_standard = augmentation_eval(response1, ground_truth_answer_standard)
    print(f"Standard Chunks - BLEU Score: {bleu_standard}")

    # Evaluate augmentation for contextual chunks
    print("\n**Evaluating Augmentation for Contextual Chunks:**")
    ground_truth_answer_contextual = "Apple's net sales grew steadily from 2019 to 2022, peaking at $394.3 billion, before a minor dip in 2023."    
    bleu_contextual = augmentation_eval(response2, ground_truth_answer_contextual)
    print(f"Contextual Chunks - BLEU Score: {bleu_contextual}")

if __name__ == "__main__":
    query = input("Enter your query: ")
    main_rag_pipeline(query)