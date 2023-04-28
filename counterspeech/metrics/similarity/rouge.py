from typing import final

import evaluate

from ..metric_base import Metric


@final
class ROUGE(Metric):
    def __init__(self, rouge_type: str = "rouge2") -> None:
        super().__init__()

        if rouge_type not in ["rouge1", "rouge2", "rougeL"]:
            raise ValueError(
                f"rouge_type must be one of ['rouge1', 'rouge2', 'rougeL'], "
                f"got {rouge_type}"
            )
        self.rouge_type = rouge_type
        self.rouge = evaluate.load("rouge")

    @property
    def name(self) -> str:
        return f"ROUGE-{self.rouge_type[5:]}"

    def calculate_score(
        self, predicted_labels: list[str], reference_labels: list[str]
    ) -> float:
        results = self.rouge.compute(
            predictions=predicted_labels,
            references=reference_labels,
            rouge_types=[self.rouge_type],
        )

        if self.rouge_type == "rougeL":
            self.score = results["rougeL"].mid.fmeasure

        else:
            self.score = results[self.rouge_type].mid.fmeasure

        return self.score
