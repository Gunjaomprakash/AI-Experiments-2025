# API Call Rules – Kodee's Farm Simulation

This document outlines the structure and behavior of API calls for Kodee's Farm, particularly with respect to toggles like Agent Mode, Image Upload, and RAG (Retrieval Augmented Generation).

## 🔘 UI Toggles Affecting API Behavior

- **Agent Mode** (on/off)
- **Image Upload** (true/false)
- **Enable RAG** (true/false)

---

## 🛰️ When Agent Mode is OFF

The API is invoked **once** with the following request structure:

### 📤 Request Payload
```json
{
  "agentMode": false,
  "isImageUploaded": true,
  "imageFile": "blob | URL | base64",
  "envValues": {
    "temperature": 92,
    "humidity": 20,
    ...
  },
  "isRAGenabled": true,
  "userPrompt": "What is wrong with Field A?"
}
```

### 📥 API Response
```json
{
  "toolList": [
    { "tool": "water_field", "timestamp": 0 },
    { "tool": "apply_pesticide", "timestamp": 4 }
  ],
  "thoughts": [
    { "text": "Humidity is critically low.", "timestamp": 1 },
    { "text": "Signs of fungal infection detected.", "timestamp": 3 }
  ],
  "finalMessage": "Field A has been stabilized after irrigation and pesticide treatment."
}
```

### ⏱️ Simulation Playback
Once a new response is received (e.g., after user submits a prompt), the **thoughts section resets** to show fresh reasoning for the current context. However, **tool calls remain cumulative**, preserving the full sequence of executed actions. Simulation playback begins by rendering entries according to timestamps:
- If a **thought** arrives first → it gets added to the "Thinking" panel.
- If a **tool call** arrives → it gets added to the tool chain.

This continues until all entries have been displayed in sync.

---

## 🟢 When Agent Mode is ON

Agent monitoring is **continuous**, with an automatic API call **every 30 seconds**.

- The structure of the request remains similar
- Backend may return partial updates (diff-only or progressive steps)
- Frontend handles continuous playback and UI refresh

---

## 👩‍💻 Developer Notes for Copilot
- This file serves as a functional reference for request/response structure
- Use this structure when creating API handler interfaces, simulation controllers, or mock responses
- All simulations must follow timestamp-based execution
- Toggle states must be wired to API trigger logic

---

> Last updated: April 2025

