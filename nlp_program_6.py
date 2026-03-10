# IMPORTS
import torch
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# GPT-2 TEXT GENERATION
print("Loading GPT-2...")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

# FIX FOR JUPYTER ERROR
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model.config.pad_token_id = gpt_model.config.eos_token_id

generator = pipeline(
    "text-generation",
    model=gpt_model,
    tokenizer=gpt_tokenizer
)

prompt = "Artificial Intelligence will transform research by"

print("\nGenerated Text:\n")

result = generator(
    prompt,
    max_length=80,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7
)

print(result[0]['generated_text'])

# T5 PARAPHRASING
print("\nLoading T5...")

t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

def paraphrase(sentence):

    text = "paraphrase: " + sentence + " </s>"

    encoding = t5_tokenizer.encode_plus(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = t5_model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=60,
        num_beams=4,
        temperature=1.5,
        early_stopping=True
    )

    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# TEST PARAPHRASING
sentence = "Deep learning improves prediction accuracy."

print("\nOriginal:")
print(sentence)

print("\nParaphrased:")
print(paraphrase(sentence))
