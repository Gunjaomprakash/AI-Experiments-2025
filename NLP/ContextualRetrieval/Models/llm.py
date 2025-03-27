from openai import OpenAI
import os

class LLM:
    def __init__(self, model_name, api_key=None, base_url=None):
        self.model = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
