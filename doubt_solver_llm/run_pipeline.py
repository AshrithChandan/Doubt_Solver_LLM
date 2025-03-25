# run_pipeline.py
import subprocess

print("\n🔍 [1/4] Scraping Google for sample questions...")
subprocess.run(["python", "scripts/scrape_google.py"], check=True)

print("\n🧹 [2/4] Cleaning and preparing the dataset...")
subprocess.run(["python", "scripts/prepare_dataset.py"], check=True)

print("\n🧠 [3/4] Training LLM with versioning...")
subprocess.run(["python", "scripts/train_with_versioning.py"], check=True)

print("\n🤖 [4/4] Launching local web interface...")
subprocess.run(["python", "inference/server.py"], check=True)

print("\n✅ All steps completed! Your LLM is ready to use via Gradio UI.")
