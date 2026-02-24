from preprocessing.clean_text import clean_text, get_sentences
from summarizers.tfidf import tfidf_summary
from summarizers.textrank import textrank_summary
from summarizers.transformer import transformer_summary
from evaluation.rouge_eval import evaluate_summary

# Read input text
with open("data/sample_article.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Preprocess
cleaned_text = clean_text(text)
sentences = get_sentences(cleaned_text)

print("\n==============================")
print("TF-IDF SUMMARY")
print("==============================")
tfidf_result = " ".join(tfidf_summary(sentences))
print(tfidf_result)

print("\n==============================")
print("TEXTRANK SUMMARY")
print("==============================")
textrank_result = " ".join(textrank_summary(sentences))
print(textrank_result)

print("\n==============================")
print("TRANSFORMER SUMMARY")
print("==============================")
transformer_result = transformer_summary(text)
print(transformer_result)

print("\n==============================")
print("ROUGE SCORES (Transformer vs Original)")
print("==============================")
scores = evaluate_summary(text, transformer_result)
print(scores)
from preprocessing.clean_text import get_sentences
