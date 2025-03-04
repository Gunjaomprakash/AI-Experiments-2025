import json
from ftfy import fix_text
from pathlib import Path

# Paths
OUTPUT_DIR = Path("processed")
INPUT_JSON = OUTPUT_DIR / "all_metadata.json"
OUTPUT_JSON = OUTPUT_DIR / "all_metadata_cleaned.json"

def clean_text_fields(data):
    """ Recursively cleans text fields inside the JSON structure """
    if isinstance(data, dict):
        return {key: clean_text_fields(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_text_fields(item) for item in data]
    elif isinstance(data, str):
        return fix_text(data)  # Apply ftfy to clean only strings
    return data

def process_json():
    """ Loads, cleans, and saves the metadata JSON """
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = clean_text_fields(data)  # Apply cleaning

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Cleaned JSON saved as: {OUTPUT_JSON}")

if __name__ == "__main__":
    process_json()