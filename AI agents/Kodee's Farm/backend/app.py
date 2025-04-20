from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from .chromadb.setup_chroma_db import db, queryKodeeMemories
from .config.system_prompt import generate_system_instruction
from google import genai
from google.genai import types
from .config.state import (
    fields,
    update_fields,
    thoughts_list,
    tools_list,
    field_snapshots,
    reset_simulation,
)
import re

from .tools.tools import (
    record_execution,
    start_irrigation,
    toggle_shade,
    trigger_fungicide_spray,
    boost_fertilizer,
    trigger_pesticide_spray,
    emergency_cooling,
    humidify_field,
    soil_recovery_treatment,
)

from .tools.ext_tools import (
    google_search,
    image_analysis,
    demo_prompt,
)

import os
import json

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
CORS(app)

# Define upload folder relative to the app.py file's directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Enable auto-reload for Flask when there are changes
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# Updated optimal ranges to match frontend values
optimal_ranges = {
    "temperature": [60, 80],  # °F
    "humidity": [40, 70],     # % (updated from 60 to 70 to match frontend)
    "soil_fertility": [60, 100],  # %
    "disease": [0, 25],       # % severity (updated from 30 to 25 to make visual feedback more sensitive)
    "rain_forecast": [20, 80], # % (added for completeness)
    "heat_wave": [0, 40],     # % (added for completeness)
}


@app.route("/api/agent", methods=["POST"])
def handle_agent_request():
    reset_simulation()
    # 1. Handle uploaded image file (if provided)
    uploaded_image = request.files.get("image")
    image_filename = None

    if uploaded_image and uploaded_image.filename != "":
        image_filename = secure_filename(uploaded_image.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
        uploaded_image.save(image_path)
        print(f"Image saved to {image_path}")
    else:
        print("No image uploaded.")

    # 2. Extract form fields
    user_message = request.form.get("userMessage", "")
    active_toggle = request.form.get("activeToggle", "user")
    attachment_enabled = request.form.get("attachmentEnabled", "false") == "true"
    # print(f"[image filename]: {image_filename}")
    try:
        received_fields = json.loads(request.form.get("fields", "[]"))
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in fields"}), 400

    update_fields(received_fields)
    # When handling a request
    agent_mode = active_toggle

    system_instruction = generate_system_instruction(
        fields=fields,
        optimal_ranges=optimal_ranges,
        uploaded_image_filename=image_filename,  # Pass the filename string
        attachment_enabled=attachment_enabled,
        agent_mode=agent_mode,
    )
    config = {
        "tools": [
            record_execution,
            start_irrigation,
            toggle_shade,
            image_analysis,
            # Only include queryKodeeMemories if attachment_enabled is True
            *([queryKodeeMemories] if attachment_enabled else []),
            boost_fertilizer,
            trigger_fungicide_spray,
            trigger_pesticide_spray,
            emergency_cooling,
            humidify_field,
            google_search,
            soil_recovery_treatment,
        ],
        "temperature": 0.2,
        "system_instruction": system_instruction,
    }

    # Check if this is a debug/demo request with a specific test scenario
    is_demo_mode = "demo" in user_message.lower() and any(crop in user_message.lower() for crop in ["banana", "corn", "tomato"])
    
    if is_demo_mode:
        # Extract the crop type from the user message
        crop_type = "banana"  # Default
        if "corn" in user_message.lower():
            crop_type = "corn"
        elif "tomato" in user_message.lower():
            crop_type = "tomato"
            
        print(f"Running in demo mode with crop: {crop_type}")
        
        # Manually create thoughts for the demo scenario
        demo_data = demo_prompt(crop_type)
        
        # Record thoughts only - tools will self-register when called
        record_execution("Analyzing the uploaded image", "")
        record_execution(f"Found in image: {demo_data['image_analysis'][:100]}...", "")
        record_execution(f"Researching growing conditions in Chicago", "")
        record_execution(f"Growing conditions: {demo_data['growing_conditions'][:100]}...", "")
        record_execution(f"Analyzing field metrics for {crop_type} cultivation", "")
        
        # Simulate tool calls based on scenario - tools will self-record
        if crop_type == "banana":
            toggle_shade(1)
            record_execution("Applied shade to reduce temperature", "")
            humidify_field(1)
            record_execution("Humidifying field to increase moisture levels", "")
        
        trigger_fungicide_spray(1)
        record_execution("Applied fungicide treatment to control disease", "")
        boost_fertilizer(1)
        record_execution("Boosted soil fertility with appropriate nutrients", "")
        
        # Use the pre-formatted final message
        raw_text = demo_data["final_message"]
        
        # Skip the normal API call
    else:
        # Normal flow with Gemini API
        chat = client.chats.create(model="gemini-2.0-flash", config=config)
        response = chat.send_message(user_message)

        # Handle response text - check if the response has the expected structure
        raw_text = ""
        if hasattr(response, "text"):
            raw_text = response.text or ""
            print(f"Raw response length: {len(raw_text)}")
            if not raw_text:
                print("Warning: Empty response.text received from Gemini")
        else:
            print(f"Warning: response object has no 'text' attribute: {type(response)}")
            # Try to extract text from other response formats
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                raw_text += part.text
                                print(f"Extracted text from candidates: {len(raw_text)} chars")

        # If raw_text is still empty, use a fallback message from thoughts_list
        if not raw_text and thoughts_list:
            fallback_message = "I've processed your request, but encountered an issue with formatting my response."
            # Create a manual FINAL_MESSAGE to ensure frontend gets a response
            raw_text = f"""
            {fallback_message}
            
            FINAL_MESSAGE: {{
                "summary": "{fallback_message}"
            }}
            """
            print("Using fallback message due to empty response")

    # Updated regex to handle both formats with or without triple backticks
    match = re.search(
        r"FINAL_MESSAGE:\s*(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})",
        raw_text,
        re.DOTALL,
    )

    if match:
        try:
            final_message_dict = json.loads(match.group(1))
            final_message = final_message_dict.get(
                "summary", "No final summary provided."
            )
        except json.JSONDecodeError:
            final_message = "Final message format error."
            print(f"JSON parse error. Captured text was: {match.group(1)}")
    else:
        # If no FINAL_MESSAGE found but we have raw text, create one from the raw text
        if raw_text:
            # Use up to 200 characters from the response as the final message
            truncated_text = raw_text[:200] + ("..." if len(raw_text) > 200 else "")
            final_message = f"Response (no formatted final message found): {truncated_text}"
        else:
            final_message = "No final message block found."
        
        # Debug: Print diagnostic information
        print(f"No FINAL_MESSAGE found. Response starts with: {raw_text[:100] if raw_text else 'EMPTY'}")
        print(
            f"Response ends with: {raw_text[-200:] if raw_text and len(raw_text) > 200 else raw_text}"
        )

    response_message = {
        "thoughtsList": thoughts_list,
        "toolList": tools_list,
        "fieldsSnapshots": field_snapshots,
        "finalMessage": final_message,
        "UploadedImage": image_filename,
    }
    print(f"Response keys: {response_message.keys()}")
    return response_message, 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
