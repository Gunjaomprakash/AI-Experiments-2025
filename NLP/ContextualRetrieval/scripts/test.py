import os
import json

DATA_DIR = os.getenv("DATA_DIR", "data/processed")
STANDARD_FILE = DATA_DIR + "/standard_items.json"

with open(STANDARD_FILE, 'r') as f:
    standard_items = json.load(f)
    print(standard_items[0])