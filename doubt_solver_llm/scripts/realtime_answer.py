# scripts/realtime_answer.py
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# === 1. Use DuckDuckGo for Search ===
from duckduckgo_search import DDGS


def search_google_snippets(query, max_results=3):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        snippets = [r["body"] for r in results if "body" in r]
        return " ".join(snippets)


# === 2. Load Latest Fine-Tuned Local Model ===
def get_latest_model_path():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    folders = [f for f in os.listdir(base_dir) if f.startswith("fine_tuned_model")]
    if not folders:
        raise FileNotFoundError(f"No fine-tuned models found in {base_dir}")
    latest = sorted(folders)[-1]
    return os.path.join(base_dir, latest)

MODEL_PATH = get_latest_model_path()

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    truncation=True,
    pad_token_id=tokenizer.eos_token_id  # ✅ Fix padding edge case
)

# === 3. Answer Generator ===
def generate_answer(question):
    context = search_google_snippets(question)[:400]  # limit context size
    print("\n📄 Context Used:\n", context)
    prompt = f"""

You are a helpful assistant. Use the context below to answer the question clearly and briefly.

### Context:
{context}

### Question:
{question}

### Answer:
"""

    result = qa_pipeline(prompt, max_length=512, do_sample=True, temperature=0.5)[0]['generated_text']
    return result.split("### Answer:")[-1].strip()

# === 4. Interactive CLI ===
if __name__ == "__main__":
    print("🔍 RealTime AnswerGPT (Local LLM + Web Search)")
    while True:
        user_input = input("\nQ: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = generate_answer(user_input)
        print("A:", answer)
