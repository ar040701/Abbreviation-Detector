import gradio as gr
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from datetime import datetime
import json
import os

# Define your label names in the correct order
label_list = ['O', 'B-AC', 'B-LF', 'I-LF']  # Update if your label set is different

model_path = os.path.abspath("../daberta_token_classifier")
# Load your saved model
model = AutoModelForTokenClassification.from_pretrained("daberta_token_classifier")
tokenizer = AutoTokenizer.from_pretrained("daberta_token_classifier")

def predict(text):
    # Preprocess
    tokens = text.split()
    inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    labels = [label_list[pred] for pred in predictions]

    # Log for monitoring (optional)
    log = {
        "timestamp": datetime.now().isoformat(),
        "input": tokens,
        "prediction": labels
    }
    with open("logs.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")

    # Return prediction
    return "\n".join(f"{tok} → {lab}" for tok, lab in zip(tokens, labels))

# Gradio UI
gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Abbreviation & Long Form Tagger (DaBERTa)",
    description="Paste biomedical text to detect abbreviations (AC) and long forms (LF)."
).launch()