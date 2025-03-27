from evaluation.evaluation_rag import retrieval_eval, augmentation_eval, generation_eval
import sys
import os

# Add the path to the modeling directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modeling'))

# Add the path to the MODELS directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import retrieve
import generate

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
    retrieved_chunks = retrieve.retrieve_chunks(query,openai_embed_llm, top_k=topk)

    print("\nStep 2: Evaluating Retrieval (Recall@K, MRR, NDCG)...")
    top_5_chunks = retrieval_eval.evaluate_retrieval(retrieved_chunks)

    print("\nStep 3: Running Augmentation Evaluation...")
    filtered_chunks = augmentation_eval.evaluate_augmentation(top_5_chunks)
    
    standard_chunks = retrieved_chunks["standard"]
    contextual_chunks = retrieved_chunks["contextual"]
    
    print("\n**Retrieved Standard Chunks:**")
    for chunk in standard_chunks:
        print(chunk["source_doc"])
        
        
    print("\n**Retrieved Contextual Chunks:**")
    for chunk in contextual_chunks:
        print(chunk["source_doc"])

    print("\nStep 4.1: Generating Answer Using the Top 5 standard Chunks...")
    response1 = generate.generate_response(query, standard_chunks, deepseek_llm)
    
    print(response1)
    
    print("\nStep 4.2: Generating Answer Using the Top 5 Contextual Chunks...")
    response2 = generate.generate_response(query, contextual_chunks, deepseek_llm)

    print(response2)
    # print("\nStep 5: Evaluating Generated Answer...")
    # final_evaluation = generation_eval.evaluate_generation(response)

    # print("\n**Final Evaluation Output:**")
    # print(final_evaluation)

    # return final_evaluation

if __name__ == "__main__":
    query = input("Enter your query: ")
    main_rag_pipeline(query)