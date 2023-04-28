from typing import final

from nltk import download, word_tokenize
from nltk.translate.bleu_score import corpus_bleu

from ..metric_base import Metric


@final
class BLEU(Metric):
    """Evaluates the similarity between predicted
    and reference labels using the BLEU score."""

    def __init__(self, n_gram: int = 4) -> None:
        super().__init__()
        assert n_gram > 0 and n_gram < 5, "n_gram must be in [1, 4]"
        download("punkt")
        self.n_gram = n_gram

    @property
    def name(self) -> str:
        return f"BLEU-{self.n_gram}"

    def calculate_score(
        self, predicted_labels: list[str], reference_labels: list[list[str]]
    ) -> float:
        assert len(predicted_labels) == len(
            reference_labels
        ), "Length of predicted_labels and reference_labels must be equal"

        self.score = corpus_bleu(
            [
                [word_tokenize(label) for label in reflist]
                for reflist in reference_labels
            ],
            [word_tokenize(label) for label in predicted_labels],
            weights=[1 / self.n_gram for _ in range(self.n_gram)],
        )

        return self.score
