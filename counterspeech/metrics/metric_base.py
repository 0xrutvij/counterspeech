from abc import ABC
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

    @property
    def name(self) -> str:
        raise NotImplementedError("Must be implemented by subclass.")

    def __str__(self) -> str:
        return f"{self.name:<16}: {self.score:>6.3f}"
