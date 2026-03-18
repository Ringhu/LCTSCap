"""Standard text generation metrics: BLEU, ROUGE, METEOR, BERTScore."""

from typing import Dict, List

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score as _meteor_score
from rouge_score import rouge_scorer

# Ensure NLTK data is available
for _pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenization."""
    return text.lower().split()


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU-1/2/3/4 using NLTK corpus_bleu with smoothing.

    Args:
        predictions: list of generated captions.
        references: list of reference captions (one reference per prediction).

    Returns:
        Dictionary with keys bleu_1, bleu_2, bleu_3, bleu_4.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    refs_tokenized = [[_tokenize(r)] for r in references]
    hyps_tokenized = [_tokenize(p) for p in predictions]

    smoother = SmoothingFunction().method1
    results: Dict[str, float] = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        results[f"bleu_{n}"] = corpus_bleu(
            refs_tokenized, hyps_tokenized, weights=weights, smoothing_function=smoother
        )
    return results


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F-measure.

    Args:
        predictions: list of generated captions.
        references: list of reference captions.

    Returns:
        Dictionary with keys rouge_1, rouge_2, rouge_l.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregated = {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        aggregated["rouge_1"] += scores["rouge1"].fmeasure
        aggregated["rouge_2"] += scores["rouge2"].fmeasure
        aggregated["rouge_l"] += scores["rougeL"].fmeasure

    n = len(predictions)
    return {k: v / n for k, v in aggregated.items()}


def compute_meteor(predictions: List[str], references: List[str]) -> float:
    """Compute average METEOR score across all prediction-reference pairs.

    Args:
        predictions: list of generated captions.
        references: list of reference captions.

    Returns:
        Average METEOR score (float).
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    total = 0.0
    for pred, ref in zip(predictions, references):
        total += _meteor_score([ref.split()], pred.split())
    return total / len(predictions)


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> Dict[str, float]:
    """Compute BERTScore precision, recall, and F1.

    Args:
        predictions: list of generated captions.
        references: list of reference captions.
        model_type: HuggingFace model name for BERTScore computation.

    Returns:
        Dictionary with keys bertscore_precision, bertscore_recall, bertscore_f1.
    """
    from bert_score import score as bert_score_fn

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    precision, recall, f1 = bert_score_fn(
        predictions, references, model_type=model_type, verbose=False
    )
    return {
        "bertscore_precision": precision.mean().item(),
        "bertscore_recall": recall.mean().item(),
        "bertscore_f1": f1.mean().item(),
    }


def compute_all_classic(
    predictions: List[str],
    references: List[str],
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
) -> Dict[str, float]:
    """Compute all classic text generation metrics.

    Runs BLEU, ROUGE, METEOR, and BERTScore and merges results into
    a single dictionary.

    Args:
        predictions: list of generated captions.
        references: list of reference captions.
        bertscore_model: HuggingFace model for BERTScore.

    Returns:
        Merged dictionary of all metric results.
    """
    results: Dict[str, float] = {}
    results.update(compute_bleu(predictions, references))
    results.update(compute_rouge(predictions, references))
    results["meteor"] = compute_meteor(predictions, references)
    results.update(compute_bertscore(predictions, references, model_type=bertscore_model))
    return results
