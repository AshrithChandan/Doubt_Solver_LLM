# scripts/scrape_google.py
import requests
from bs4 import BeautifulSoup
import json
import time
import os



HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/58.0.3029.110 Safari/537.3"
}

# Search Google and return top results' snippets
def search_google(query):
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    snippets = []

    for result in soup.select(".tF2Cxc"):
        snippet = result.select_one(".VwiC3b")
        if snippet:
            snippets.append(snippet.text)

    return " ".join(snippets[:3])  # Return top 3 snippets combined

# Generate question-answer pairs
def build_qa_pairs(queries, output_file):
    qa_pairs = []
    for question in queries:
        print(f"Searching: {question}")
        context = search_google(question)
        if context:
            qa_pairs.append({
                "prompt": question,
                "completion": context.strip()
            })
        time.sleep(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # ✅ Create the folder if missing
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(qa_pairs)} QA pairs to {output_file}")


# Example usage
if __name__ == "__main__":
    sample_questions = [
        "What is a neural network?",
        "Why is the sky blue?",
        "What is overfitting in machine learning?"
    ]
    build_qa_pairs(sample_questions, "../data/google_qa_dataset.json")
