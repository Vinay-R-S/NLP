"""
NLP Program 9: Machine Translation with Transformers
"""
from transformers import MarianMTModel, MarianTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import pipeline

def translate_with_marian(text):
    print("\nTranslating with MarianMT (English to French)...")
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded)
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text

def translate_with_pipeline(text):
    print("\nTranslating with translation pipeline...")
    # Explicitly providing the model to avoid warnings, using the MarianMT model suitable for EN-FR
    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    result = translator(text, max_length=50)
    return result[0]['translation_text']

def translate_with_mbart(text):
    print("\nTranslating with mBART-50 (English to French)...")
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    
    # Set source language
    tokenizer.src_lang = "en_XX"
    encoded = tokenizer(text, return_tensors="pt")
    
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
    )
    
    translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_texts[0]

def main():
    text1 = "Artificial Intelligence is changing the world."
    print(f"Original Text: {text1}")
    
    marian_translation = translate_with_marian(text1)
    print(f"MarianMT Translation: {marian_translation}")
    
    pipeline_translation = translate_with_pipeline(text1)
    print(f"Pipeline Translation: {pipeline_translation}")
    
    text2 = "Machine learning is powerful."
    print(f"\nOriginal Text: {text2}")
    
    mbart_translation = translate_with_mbart(text2)
    print(f"mBART Translation: {mbart_translation}")

if __name__ == "__main__":
    main()
