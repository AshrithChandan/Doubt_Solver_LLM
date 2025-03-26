🧠 doubt_solver_llm
MIT License Python Transformers

doubt_solver_llm is a full-stack local AI pipeline that lets you build, train, and run your own lightweight ChatGPT-style assistant that can answer real-time user questions using fine-tuned language models and live search (DuckDuckGo).

You can:

Fine-tune models like TinyLlama, Phi-2, or Zephyr
Automatically build datasets from Google/DuckDuckGo
Train with prompt + context + answer format
Ask questions via CLI or a Gradio Web App
🚀 Features
✅ Fine-tunes open-source LLMs (TinyLlama/Zephyr/etc.)
✅ Real-time DuckDuckGo search + context feeding
✅ Clean prompt engineering: ### Context: → ### Question: → ### Answer:
✅ Auto dataset generation using GPT-backed search
✅ Modular scripts for scraping, preprocessing, training, inference
✅ CLI + Gradio Web UI for asking questions
✅ Lightweight and local-first, works on CPU or GPU

📁 Project Structure
doubt_solver_llm/
├── data/                     # QA datasets (scraped + cleaned)
│   ├── google_qa_dataset.json
│   └── questions.json
│
├── models/                  # Fine-tuned model versions (auto-created)
│   ├── fine_tuned_model_20250325_053845/
│   └── ...
│
├── scripts/                 # Core logic scripts
│   ├── scrape_google.py          # Uses Google/DuckDuckGo to collect Q&A
│   ├── prepare_dataset.py        # Cleans scraped data
│   ├── train_model.py            # Simple static fine-tuning (not recommended)
│   ├── train_with_versioning.py  # ✅ Recommended fine-tuning script
│   └── realtime_answer.py        # Real-time search + local model answering
│
├── inference/              # Interfaces
│   ├── ask.py              # CLI Q&A tool
│   └── server.py           # Gradio web app
│
├── run_pipeline.py         # Full auto-pipeline: scrape → clean → train → serve
├── requirements.txt
└── README.md               # This file
🧪 Quickstart
1. 🧰 Install Dependencies
pip install -r requirements.txt
2. 🚀 Run Everything
python run_pipeline.py
This will:

Scrape 3 questions via DuckDuckGo
Clean and prepare the Q&A data
Fine-tune the LLM
Launch the Gradio web app
3. 🔁 Use Components Manually
python scripts/scrape_google.py         # Scrape questions
python scripts/prepare_dataset.py       # Clean data
python scripts/train_with_versioning.py # Fine-tune model
python inference/ask.py                 # Ask from terminal
python inference/server.py              # Use Gradio UI
🧠 Fine-Tuning Details
📄 Input Format
Your training data should look like:

{
  "prompt": "What is a neural network?",
  "completion": "A neural network is a series of algorithms that mimic the human brain..."
}
This gets converted into:

### Question:
What is a neural network?

### Answer:
A neural network is...
🧠 Model Used
Default: TinyLlama/TinyLlama-1.1B-Chat-v1.0
You can change this in train_with_versioning.py:

MODEL_NAME = "microsoft/phi-2"
🔁 Versioned Output
Models are saved like:

models/fine_tuned_model_YYYYMMDD_HHMMSS/
Each time you run train_with_versioning.py

🤖 Real-Time Web + CLI Search QA
CLI:
python scripts/realtime_answer.py
You can now ask:

Q: Why do cats purr?
And it will:

Search the web live with DuckDuckGo
Extract relevant snippets
Generate a local LLM-powered answer from context
Gradio UI:
python inference/server.py
Access from your browser and type questions.

📦 Dependencies
transformers
datasets
duckduckgo-search
requests
beautifulsoup4
gradio
openai
Install with:

pip install -r requirements.txt
🔐 License
MIT License — Free to use, modify, and build on.

👨‍💻 Author
Created by Ashrith Chandan
Contributions welcome! Let’s build smarter local AI together 💪
