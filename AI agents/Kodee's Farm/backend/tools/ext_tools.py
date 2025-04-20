import os
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch


client = genai.Client(api_key= os.getenv("GOOGLE_API_KEY"))

def image_analysis(img_name: str) -> dict:
    """
    Analyzes a farming-related image using Google Gemini and returns a concise description.

    Args:
        img_name (str): The name of the image file to analyze.

    Returns:
        dict: A dictionary with a 'description' key containing the analysis result.
    """
    # Correct the absolute path to avoid duplication
    upload_dir = '/Users/omprakashgunja/Documents/GitHub/AI-Experiments-2025/AI-Experiments-2025/AI agents/Kodee\'s Farm/backend/uploads'
    upload_path = os.path.join(upload_dir, img_name)
    myfile = client.files.upload(file=upload_path)
    prompt = (
        "You are an expert in agricultural image analysis. "
        "Examine the provided image and deliver a concise, structured description of what you see. "
        "Focus on identifying the crop, its health, visible issues (such as pests, diseases, or nutrient deficiencies), "
        "and any notable features relevant to farming. "
        "Respond in the format: description: {your concise analysis}"
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[myfile, prompt]
    )
    if hasattr(response, "text") and response.text:
        return {"description": response.text.strip()}
    else:
        return {"description": "No clear analysis available."}
    
def google_search(query: str) -> dict:
    """
    Uses Google Gemini's search tool to find information on the internet.

    Args:
        query (str): The search query string to look up.

    Returns:
        dict: A dictionary containing the search result text and optional grounding metadata.

    Example:
        result = google_search("current time in Chicago")
        print(result["text"])
    """
    print(f"[google_search] Query: {query}")  # Debug print
    client = genai.Client(api_key= os.getenv("GOOGLE_API_KEY"))
    search_tool = Tool(google_search=GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[search_tool],
        response_mime_type="text/plain"
    )
    prompt = (
    f"You are an expert research assistant. "
    f"Search the web and provide a concise, up-to-date, and structured answer for the following query. "
    f"Always include the most recent data available (preferably 2024 or 2025), relevant statistics, and cite your sources if possible. "
    f"If the answer is not directly available, explain what is missing and suggest how the user might find it. "
    f"Return your answer as a short summary "
    f"Query: {query}"
)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=config
    )
    metadata = None

    print(f"[google_search] Response: {response.text}")  # Debug print
    return {"text": response.text, "grounding_metadata": metadata}

# image_analysis("banana.jpeg")
# google_search("best crops to grow Chicago")