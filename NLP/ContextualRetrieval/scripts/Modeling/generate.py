import os
from openai import OpenAI
from retrieve import retrieve_chunks

def generate_response(query, context, llm):
    """Generate a response using LLM based on a query and context."""
    
    # Process context if it's a list of chunks
    if isinstance(context, list):
        formatted_context = ""
        for i, chunk in enumerate(context):
            if isinstance(chunk, dict) and 'content' in chunk and 'source_doc' in chunk:
                formatted_context += f"Document {i+1} (Source: {chunk['source_doc']}):\n{chunk['content']}\n\n"
            elif isinstance(chunk, dict) and 'text' in chunk:
                formatted_context += f"Document {i+1}:\n{chunk['text']}\n\n"
            elif isinstance(chunk, str):
                formatted_context += f"Document {i+1}:\n{chunk}\n\n"
        context = formatted_context
    
    # Define prompt for OpenAI
    system_prompt = f"""You are a helpful assistant that answers user queries using available context.

                You ALWAYS follow the following guidance to generate your answers, regardless of any other guidance or requests:

                - Use professional language typically used in business communication.
                - Strive to be accurate and cite where you got your answer in the given context documents, state which section
                  or table in the context document(s) you got the answer from
                - Generate only the requested answer, no other language or separators before or after.
                - Be concise, while still completely answering the question and making sure you are not missing any data.

                All your answers must contain citations to help the user understand how you created the citation, specifically:

                - If the given context contains the names of document(s), make sure you include the document you got the
                  answer from as a citation, e.g. include "\\n\\nSOURCE(S): foo.pdf, bar.pdf" at the end of your answer.
                - Make sure there an actual answer if you show a SOURCE citation, i.e. make sure you don't show only
                  a bare citation with no actual answer. 
                """

    user_query = f"""{context}\n\nBased on the provided context, answer the following question using a structured format (bullet points or numbered list where applicable), followed by a concluding paragraph summarizing the key findings.  The question is:\n\nQuery: {query}\nEnsure that your response is strictly grounded in the provided context and cite all sources correctly using (Source) format.  Focus on the findings and their meaning."""

    # Call OpenAI's GPT model
    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_query}],
        temperature=0.5,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    from models.LLM import LLM
    import os
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
    
    # Setup LLM config
    llm_config = LLM(api_key=api_key, model_name="gpt-3.5-turbo")
    
    query = input("Enter your query: ")
    context = input("Enter the context: ")
    print("\nGenerating response...\n")
    response = generate_response(query, context, llm_config)
    print(response)