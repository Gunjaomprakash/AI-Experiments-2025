import os
from openai import OpenAI
from retrieve import retrieve_chunks

# Load OpenAI API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# # client= OpenAI(api_key=OPENAI_API_KEY)
# client= OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# # model = "gpt-4-1106-preview"
# model = "deepseek-chat"

def generate_response(query,context,llm):
    """Retrieve relevant chunks and generate a response using OpenAI."""
    
    # Define prompt for OpenAI
    system_prompt = f"""You are a helpful assistant that answers user queries using available context.

                    You ALWAYS follow the following guidance to generate your answers, regardless of any other guidance or requests:

                    - Use professional language typically used in business communication.
                    - Strive to be accurate and cite where you got your answer in the given context documents, state which  section
                    or table in the context document(s) you got the answer from
                    - Generate only the requested answer, no other language or separators before or after.
                    - Be concise, while still completely answering the question and making sure you are not missing any data.

                    All your answers must contain citations to help the user understand how you created the citation, specifically:

                    - If the given context contains the names of document(s), make sure you include the document you got the
                    answer from as a citation, e.g. include "\\n\\nSOURCE(S): foo.pdf, bar.pdf" at the end of your answer.
                    - Make sure there an actual answer if you show a SOURCE citation, i.e. make sure you don't show only
                    a bare citation with no actual answer. 

                    """

    user_query = f""" {context}\n\nBased on the provided document context, generate a response to the following question:\n\nQuery: {query}\nEnsure that your response is strictly grounded in the provided document chunks. Cite all sources correctly."""

    
    # Call OpenAI's GPT model
    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_query}],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

# def retrieve_docs(query, top_k=3):
#     """Retrieve relevant chunks for a given query."""
#     print(f"Retrieving top {top_k} relevant chunks for: query")
    
#     retrieved_chunks = retrieve_chunks(query, top_k)
#     standard_context, contextual_context= "", ""

#     if not retrieved_chunks:
#         return "I couldn't find relevant information to answer your query."

#     return standard_context, contextual_context

if __name__ == "__main__":
    query = input("Enter your query: ")
    context = input("Enter the context: ")
    print("\nGenerating response...\n")
    response = generate_response(query,context)
    print(response)
    # standard_context, contextual_context = retrieve_docs(query, top_k=3)
    
    # standard_response = generate_response(query,standard_context, top_k=3)
    # contextual_response = generate_response(query,contextual_context, top_k=3)
    
    # print("\nStandard Response:\n")
    # print(standard_response)
    
    # print("\nContextual Response:\n")
    # print(contextual_response)