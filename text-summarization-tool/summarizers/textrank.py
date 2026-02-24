from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def textrank_summary(sentences, num_sentences=2):
    """
    Ranks sentences based on similarity (TextRank-style)
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Sentence-to-sentence similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Score sentences by how similar they are to others
    scores = similarity_matrix.sum(axis=1)

    # Pick top sentences
    top_indices = np.argsort(scores)[-num_sentences:]
    top_indices.sort()

    return [sentences[i] for i in top_indices]
