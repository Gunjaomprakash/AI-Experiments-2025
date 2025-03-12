# import modeling.retrieve as retrieve
# # import evaluation.retrieval_evaluation as retrieval_eval
# # import evaluation.augmentation_evaluation as augmentation_eval
# import modeling.generate as generate
# # import evaluation.generation_evaluation as generation_eval
import sys
import os

# Add the path to the modeling directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Modeling'))

import retrieve
import generate

def main_rag_pipeline(query):
    """
    Custom Python pipeline for RAG evaluation without LangChain overhead.
    """
    topk = 10
    print(f"\nStep 1: Retrieving Top {topk} Chunks...")
    retrieved_chunks = retrieve.retrieve_chunks(query, top_k=topk)

    # print("\nStep 2: Evaluating Retrieval (Recall@K, MRR, NDCG)...")
    # top_5_chunks = retrieval_eval.evaluate_retrieval(retrieved_chunks)

    # print("\nStep 3: Running Augmentation Evaluation...")
    # filtered_chunks = augmentation_eval.evaluate_augmentation(top_5_chunks)
    
    standard_chunks = retrieved_chunks["standard"]
    contextual_chunks = retrieved_chunks["contextual"]
    
    print("\n**Retrieved Standard Chunks:**")
    for chunk in standard_chunks:
        print(chunk["source_doc"])
        
        
    print("\n**Retrieved Contextual Chunks:**")
    for chunk in contextual_chunks:
        print(chunk["source_doc"])

    print("\nStep 4.1: Generating Answer Using the Top 5 standard Chunks...")
    response1 = generate.generate_response(query, standard_chunks)
    
    print(response1)
    
    print("\nStep 4.2: Generating Answer Using the Top 5 Contextual Chunks...")
    response2 = generate.generate_response(query, contextual_chunks)

    print(response2)
    # print("\nStep 5: Evaluating Generated Answer...")
    # final_evaluation = generation_eval.evaluate_generation(response)

    # print("\n**Final Evaluation Output:**")
    # print(final_evaluation)

    # return final_evaluation

if __name__ == "__main__":
    query = input("Enter your query: ")
    main_rag_pipeline(query)