# scripts/train_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

# Settings
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "../data/questions.json"
OUTPUT_DIR = "../models/fine_tuned_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Load dataset
dataset = load_dataset("json", data_files=DATA_PATH)

# Format prompt + answer
PROMPT_TEMPLATE = """
### Question:
{prompt}

### Answer:
{completion}
"""

def format(example):
    full_text = PROMPT_TEMPLATE.format(prompt=example["prompt"], completion=example["completion"])
    return {"text": full_text}

dataset = dataset.map(format)

# Tokenize dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    evaluation_strategy="no",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=collator,
)

# Train
trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("\n✅ Training complete. Model saved to:", OUTPUT_DIR)
