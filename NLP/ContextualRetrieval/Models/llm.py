from openai import OpenAI
import os

class LLM:
    def __init__(self, model_name=, api_key=None, base_url=None):
        self.model = model_name
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
