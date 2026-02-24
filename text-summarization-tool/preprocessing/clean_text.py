import re
import nltk
from nltk.tokenize import sent_tokenize

# Download once (safe to keep, it won’t re-download if already present)
nltk.download('punkt')
nltk.download('punkt_tab')

def clean_text(text: str) -> str:
    # Fix missing spaces after punctuation: "time.This" -> "time. This"
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)

    # Fix missing spaces between lowercase-uppercase: "AboutSuppose" -> "About Suppose"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Fix missing spaces between numbers and letters: "1.1Whats" -> "1.1 Whats"
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    # Normalize extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_sentences(text: str):
    cleaned = clean_text(text)
    sentences = sent_tokenize(cleaned)

    # Capitalize each sentence nicely
    sentences = [s.strip().capitalize() for s in sentences]

    return sentences
