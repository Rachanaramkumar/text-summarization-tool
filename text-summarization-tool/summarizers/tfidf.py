from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def tfidf_summary(sentences, num_sentences=2):
    """
    Selects top sentences based on TF-IDF scores
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Score each sentence
    scores = tfidf_matrix.sum(axis=1).A1

    # Get indices of top sentences
    top_indices = np.argsort(scores)[-num_sentences:]
    top_indices.sort()

    return [sentences[i] for i in top_indices]
