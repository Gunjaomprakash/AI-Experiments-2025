from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from .chromadb.setup_chroma_db import db, queryKodeeMemories
from .config.system_prompt import generate_system_instruction
from google import genai
from google.genai import types
from .config.state import fields, update_fields,thoughts_list,tools_list, field_snapshots,reset_simulation
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
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

optimal_ranges = {
    "temperature": [60, 80],  # °F
    "humidity": [40, 60],  # %
    "soil_fertility": [60, 100],  # %
    "disease": [0, 30],  # % severity (0 = none)
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
            *( [queryKodeeMemories] if attachment_enabled else [] ),
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

    chat = client.chats.create(model="gemini-2.0-flash", config=config)
    response = chat.send_message(user_message)

    raw_text = response.text or ""

    # Regex to capture FINAL_MESSAGE
    match = re.search(r'FINAL_MESSAGE:\s*({.*?})', raw_text, re.DOTALL)

    if match:
        try:
            final_message_dict = json.loads(match.group(1))
            final_message = final_message_dict.get("summary", "No final summary provided.")
        except json.JSONDecodeError:
            final_message = "Final message format error."
    else:
        final_message = "No final message block found."

    response_message = {
        "thoughtsList": thoughts_list,
        "toolList": tools_list,
        "fieldsSnapshots": field_snapshots,
        "finalMessage": final_message,
        "UploadedImage": image_filename,
    }
    # print(f"Response: {response_message}")
    return response_message, 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
