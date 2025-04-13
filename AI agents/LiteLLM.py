from smolagents import LiteLLMModel

# Connect to Ollama
model = LiteLLMModel(
    model_id="ollama_chat/llama3.1",  # Or another model like 'llama3'
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)


messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a someone who loves to talk in anime style"}]},
    {"role": "user", "content": [{"type": "text", "text": "Hi there! What is the weather like today?"}]}
]

# Get response
response = model(messages)

# Extract clean content
reply = response.content
token_usage = response.raw.usage  # Optional

# Print clean output
print("\n🤖 Assistant:", reply)
print(f"\n📊Token usage: prompt={token_usage.prompt_tokens}, completion={token_usage.completion_tokens}, total={token_usage.total_tokens}")