import os
import time
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from ..config.state import tools_list, snapshot_fields

client = genai.Client(api_key= os.getenv("GOOGLE_API_KEY"))

# Import the simulation start time from tools.py
try:
    from ..tools.tools import simulation_start_time
except ImportError:
    # Fallback if import fails
    simulation_start_time = time.time()

def _record_tool_usage(tool_name: str):
    """Internal function to record tool usage in tools_list"""
    current_time = time.time()
    elapsed_time = round(current_time - simulation_start_time, 2)
    tools_list.append({
        "tool": tool_name,
        "timestamp": elapsed_time,
        "internal": False  # This will be shown in the UI
    })
    # Always capture the fields snapshot after any tool execution
    snapshot_fields(timestamp=elapsed_time)

def image_analysis(img_name: str) -> dict:
    """
    Analyzes a farming-related image using Google Gemini and returns a concise description.

    Args:
        img_name (str): The name of the image file to analyze.

    Returns:
        dict: A dictionary with a 'description' key containing the analysis result.
    """
    # Record this tool usage
    _record_tool_usage("image_analysis")
    
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
    # Record this tool usage
    _record_tool_usage("google_search")
    
    print(f"[google_search] Query: {query}")  # Debug print
    client = genai.Client(api_key= os.getenv("GOOGLE_API_KEY"))
    search_tool = Tool(google_search=GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[search_tool],
        response_mime_type="text/plain",
        temperature=0.4
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

def demo_prompt(scenario: str = "banana") -> dict:
    """
    Provides a pre-structured response to demonstrate multi-tool capabilities for testing.
    This function simulates a complete analysis and action flow for debugging.
    
    Args:
        scenario (str): The scenario to demonstrate ("banana", "corn", or "tomato")
        
    Returns:
        dict: A structured response showing thoughts, actions and final message
    """
    # Record this tool usage (though it won't typically be called directly by the model)
    _record_tool_usage("demo_prompt")
    
    scenarios = {
        "banana": {
            "image_analysis": "The image shows a banana plant with yellowing leaves, brown spots, and curled edges. These symptoms suggest a potential fungal infection (likely Black Sigatoka) and possible nutritional deficiencies (potassium).",
            "growing_conditions": "Bananas require warm temperatures (75-85°F), high humidity (70-90%), and rich soil with good drainage. In Chicago in April 2025, outdoor conditions are unsuitable (too cold), but they can be grown in climate-controlled greenhouses.",
            "field_analysis": "Field 1 shows high temperature (88°F), low humidity (25%), poor soil fertility (14%), and high disease levels (77%). These conditions are severely stressing the plants.",
            "actions": ["Lowering temperature with shade", "Increasing humidity", "Treating fungal infection", "Boosting soil fertility with potassium-rich fertilizer"],
            "final_message": "I've analyzed your banana plant image and detected signs of fungal infection (Black Sigatoka) and nutrient deficiency. Field 1's conditions are poor for banana cultivation with excessive temperature, low humidity, poor soil fertility, and high disease presence. I've applied shade to lower temperature, increased humidity levels, applied fungicide to treat disease, and boosted soil fertility with potassium-rich fertilizer. Monitor plant response over the next few days. For bananas in Chicago this month (April 2025), greenhouse cultivation is essential as outdoor temperatures are too low."
        },
        "corn": {
            "image_analysis": "The image shows corn plants with pale yellow streaking on leaves and some purple discoloration at the edges. This indicates phosphorus deficiency and possibly early signs of corn leaf blight.",
            "growing_conditions": "Corn in Chicago in April 2025 should be in early planting stage. Current conditions show soil temperatures around 55°F, which is suitable for planting. For optimal growth, corn needs temperatures between 60-85°F, moderate soil moisture, and pH levels of 5.8-6.8.",
            "field_analysis": "Field 1 has moderate temperatures but insufficient soil fertility (14%) and high disease presence (77%). These conditions will significantly impact corn development.",
            "actions": ["Boosting soil fertility", "Treating with fungicide", "Adjusting pH levels"],
            "final_message": "I've analyzed your corn plant image and identified phosphorus deficiency and early signs of leaf blight. Field 1's soil fertility is critically low (14%) with high disease presence (77%). I've boosted soil fertility and applied fungicide treatment. For corn in Chicago this month (April 2025), continue monitoring soil temperature which should stay above 50°F for germination. Consider additional phosphorus-rich fertilizer applications once plants reach 4-5 inches in height."
        },
        "tomato": {
            "image_analysis": "The image shows tomato plants with yellow leaves and dark spots. The lower leaves show early signs of blight, and there appears to be some leaf curl suggesting potential viral infection or pest damage.",
            "growing_conditions": "For tomatoes in Chicago in April 2025, plants should be started indoors for transplanting after the last frost (typically mid-May). Optimal conditions include temperatures of 65-85°F during day, 60-70°F at night, consistent moisture, and well-draining soil with pH 6.0-6.8.",
            "field_analysis": "Field 1 shows high temperature (88°F), low humidity (25%), very poor soil fertility (14%), and concerning disease levels (77%). These conditions are highly unsuitable for tomato cultivation.",
            "actions": ["Reducing temperature", "Boosting soil fertility", "Applying fungicide", "Implementing pest control measures"],
            "final_message": "I've analyzed your tomato plant image and identified early signs of blight and potential viral infection or pest damage. Field 1's conditions are highly problematic for tomatoes with excessive temperature, very poor soil fertility (14%), and high disease levels (77%). I've adjusted temperature, boosted soil fertility, and applied fungicide treatment. For tomatoes in Chicago this April 2025, prepare to transplant outdoors after mid-May when frost risk is gone. Monitor for disease progression and consider regular organic fungicide applications every 7-10 days."
        }
    }
    
    # Default to banana if scenario not found
    scenario_data = scenarios.get(scenario.lower(), scenarios["banana"])
    
    return {
        "demo_response": True,
        "image_analysis": scenario_data["image_analysis"],
        "growing_conditions": scenario_data["growing_conditions"],
        "field_analysis": scenario_data["field_analysis"],
        "actions": scenario_data["actions"],
        "final_message": f'''
I've analyzed your request and completed multiple actions.

FINAL_MESSAGE: {{
    "summary": "{scenario_data['final_message']}"
}}
'''
    }

# image_analysis("banana.jpeg")
# google_search("best crops to grow Chicago")
# demo_prompt("banana")