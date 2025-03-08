import os
from openai import OpenAI
from retrieve import retrieve_chunks

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY)

def generate_response(query,context, top_k=3):
    """Retrieve relevant chunks and generate a response using OpenAI."""
    
    # Define prompt for OpenAI
    prompt = f"""You are a highly accurate AI assistant. Use the provided context to answer the user's question. Your response must:
        - Extract only relevant facts from the given sources.
        - Clearly cite the sources using the format: `(SOURCE: [Document Name], Page [Number])`.
        - Structure the response in a clear, readable format (e.g., bullet points for multiple data points).
        - Avoid speculation—if the context lacks information, state that no relevant data was found.

    ### Context:
    {context}

    ### Instructions:
        - If multiple sources provide information, list them separately.
        - For numerical data, clearly associate it with the correct source.
        - If no relevant data is available, say: `"The available documents do not contain relevant information on this topic."`

    ### Question:
    {query}

    ### Answer:
    """

    # Call OpenAI's GPT model
    response = client_openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

def retrieve_docs(query, top_k=3):
    """Retrieve relevant chunks for a given query."""
    print(f"Retrieving top {top_k} relevant chunks for: query")
    
    retrieved_chunks = retrieve_chunks(query, top_k)
    standard_context, contextual_context= "", ""
    print("\nStandard Chunks:")
    if not retrieved_chunks:
        return "I couldn't find relevant information to answer your query."
    
    print(retrieved_chunks)


    return standard_context, contextual_context

if __name__ == "__main__":
    query = input("Enter your query: ")
    standard_context, contextual_context = retrieve_docs(query, top_k=3)
    
    standard_response = generate_response(query,standard_context, top_k=3)
    contextual_response = generate_response(query,contextual_context, top_k=3)
    
    print("\nStandard Response:\n")
    print(standard_response)
    
    print("\nContextual Response:\n")
    print(contextual_response)