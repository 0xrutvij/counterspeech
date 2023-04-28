from abc import ABC, abstractmethod
from typing import Optional


class Metric(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._score: Optional[float] = None
        self.individual_scores: Optional[list[float]] = None

    @property
    def score(self) -> float:
        if self._score is None:
            raise ValueError("Score not set. Call `calculate_score()` first.")
        return self._score

    @score.setter
    def score(self, value: float) -> None:
        self._score = value

    @abstractmethod
    @property
    def name(self) -> str:
        ...

    def __str__(self) -> str:
        return f"{self.name:<16}: {self.score:>6.3f}"

    @abstractmethod
    def calculate_score(
        self,
        predicted_labels: list[str],
        reference_labels: list[str],
    ) -> float:
        ...
