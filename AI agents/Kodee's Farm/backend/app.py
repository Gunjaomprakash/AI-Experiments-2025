from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/api/agent', methods=['POST'])
def handle_agent_request():
    # 1. Handle uploaded image file (if provided)
    uploaded_image = request.files.get('image')
    image_filename = None

    if uploaded_image and uploaded_image.filename != '':
        image_filename = secure_filename(uploaded_image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        uploaded_image.save(image_path)
        print(f"Image saved to {image_path}")
    else:
        print("No image uploaded.")

    # 2. Extract form fields
    user_message = request.form.get('userMessage', '')
    active_toggle = request.form.get('activeToggle', 'user')
    attachment_enabled = request.form.get('attachmentEnabled', 'false') == 'true'

    try:
        fields = json.loads(request.form.get('fields', '[]'))
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON in fields'}), 400

    # 3. Log received data
    print("== Incoming /api/agent request ==")
    print("User message:", user_message)
    print("Active toggle:", active_toggle)
    print("Attachment (RAG) enabled:", attachment_enabled)
    print("Fields:", fields)
    print("Uploaded image:", image_filename)

    # 4. Run your AI agent logic here (updated response with timestamps)
    dummy_thoughts = [
        {"text": "Analyzing weather metrics...", "timestamp": 0},
        {"text": "Preparing crop treatment plan.", "timestamp": 3}
    ]
    dummy_tools = [
        {"tool": "trigger_irrigation", "timestamp": 1},
        {"tool": "adjust_fertilizer", "timestamp": 2},
        {"tool": "adjust_fertilizer", "timestamp": 4}
    ]
    dummy_final = "Irrigation initiated and soil health balanced for Field 1. 🌱"

    return jsonify({
        "thoughtsList": dummy_thoughts,
        "toolList": dummy_tools,
        "finalMessage": dummy_final,
        "imageUsed": image_filename  # Optional for frontend preview/debug
    }), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)