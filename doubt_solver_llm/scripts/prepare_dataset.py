# scripts/prepare_dataset.py
import json
import random

INPUT_FILE = "../data/google_qa_dataset.json"
OUTPUT_FILE = "../data/questions.json"

# Load scraped Q&A data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Optional: shuffle and clean
random.shuffle(data)
cleaned_data = []

for item in data:
    prompt = item.get("prompt", "").strip()
    completion = item.get("completion", "").strip()
    if prompt and completion:
        cleaned_data.append({
            "prompt": prompt,
            "completion": completion
        })

# Save cleaned dataset
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

print(f"✅ Prepared dataset with {len(cleaned_data)} Q&A pairs → {OUTPUT_FILE}")
