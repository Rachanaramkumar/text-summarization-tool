from rouge_score import rouge_scorer

def evaluate_summary(reference, summary):
    """
    Evaluates summary quality using ROUGE metrics
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    scores = scorer.score(reference, summary)
    return scores
