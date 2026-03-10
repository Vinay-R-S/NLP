# IMPORT LIBRARIES
import os
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
dataset

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
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# LOAD MODEL
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# METRICS FUNCTION
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

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
    logging_dir="./logs",
    logging_steps=100
)

# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)

# TRAIN MODEL
trainer.train()

# EVALUATE MODEL
trainer.evaluate()
