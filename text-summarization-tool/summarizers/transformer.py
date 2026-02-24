from transformers import pipeline

# Load a pre-trained summarization model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def transformer_summary(text):
    """
    Generates an abstractive summary using a Transformer model
    """
    summary = summarizer(
        text,
        max_length=120,
        min_length=40,
        do_sample=False
    )
    return summary[0]['summary_text']
