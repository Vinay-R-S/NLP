# IMPORT LIBRARIES
import os
import torch
import numpy as np

from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# FIX HUGGINGFACE CACHE ERROR (OPTIONAL BUT RECOMMENDED)
os.environ["HF_DATASETS_CACHE"] = "./hf_cache"

# LOAD DATASET (IMDb)
print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

# INITIALIZE TOKENIZER
print("Initializing tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# TOKENIZATION FUNCTION
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True
    )

# TOKENIZE DATASET (CACHE DISABLED)
print("Tokenizing dataset...")
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    load_from_cache_file=False
)

# SET FORMAT FOR PYTORCH
tokenized_datasets.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'label']
)

# LOAD MODEL
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# METRICS FUNCTION
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# TRAINING CONFIGURATION
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="no",  # avoids file overwrite
    logging_dir="./logs"
)

# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(1000)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# TRAIN MODEL
print("Training started...")
trainer.train()

# EVALUATE MODEL
print("Evaluating...")
results = trainer.evaluate()
print("Evaluation Results:", results)

# SAVE MODEL SAFELY
model.save_pretrained("bert-imdb-model")
tokenizer.save_pretrained("bert-imdb-model")

# PREDICTION FUNCTION
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits).item()
    return "Positive Review 😊" if predicted_class_id == 1 else "Negative Review 😞"

# TEST PREDICTIONS
print("\nSample Predictions:")
print("1.", predict("This movie was amazing and full of emotions!"))
print("2.", predict("This was the worst acting I have ever seen."))
