"""
NLP Program 8: Text Summarization and Web Scraping
"""
import re
import requests
from bs4 import BeautifulSoup
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

def fetch_and_clean_text(url):
    print(f"Fetching content from: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract paragraphs
    paragraphs = soup.find_all("p")
    text = " ".join([p.get_text() for p in paragraphs])
    
    # Clean text
    clean_text = re.sub(r'\s+', ' ', text).strip()
    return clean_text

def summarize_with_bart(text):
    print("\nSummarizing with BART...")
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # BART models typically handle longer sequences, up to 1024 tokens
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    
    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4, # Use beam search for better quality
        max_length=150, # Max length of the generated summary
        min_length=30,  # Min length of the generated summary
        early_stopping=True
    )
    
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

def summarize_with_t5(text):
    print("\nSummarizing with T5...")
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=150)
    summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary_text

def main():
    url = "https://rvu.edu.in/rvu-at-a-glance/"
    clean_text = fetch_and_clean_text(url)
    print("Preview of clean text (first 500 chars):")
    print(clean_text[:500] + "...\n")
    
    bart_summary = summarize_with_bart(clean_text)
    print("--- BART Summary ---")
    print(bart_summary)
    
    t5_summary = summarize_with_t5(clean_text)
    print("--- T5 Summary ---")
    print(t5_summary)

if __name__ == "__main__":
    main()
