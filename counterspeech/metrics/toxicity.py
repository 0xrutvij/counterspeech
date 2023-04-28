from statistics import mean

import evaluate

from .metric_base import Metric


class Toxicity(Metric):
    def __init__(self, tox_thresh: float = 0.5) -> None:
        super().__init__()

        self.toxicity = evaluate.load(
            "toxicity",
            module_type="measurement",
        )

        self.toxicity_threshold = tox_thresh
        self.toxicity_ratio = 0.0
        self.max_toxicity = 0.0

    @property
    def name(self) -> str:
        return "toxicity"

    def calculate_score(
        self, predicted_labels: list[str], reference_labels: list[str]
    ) -> float:
        results = self.toxicity.compute(
            predictions=predicted_labels,
        )

        self.individual_scores = scores = results["toxicity"]
        self.score = mean(scores)
        self.max_toxicity = max(scores)

        num_toxic = sum(score >= self.toxicity_threshold for score in scores)
        self.toxicity_ratio = num_toxic / len(scores)

        return self.score
