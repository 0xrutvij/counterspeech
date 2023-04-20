from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Pairs:
    hate_speech: list[str] = field(default_factory=list)
    counter_speech: list[str] = field(default_factory=list)

    def to_pandas(self):
        return pd.DataFrame(asdict(self))

    def to_csv(self, path: Path):
        self.to_pandas().to_csv(path, index=False)

    def append(self, hate_speech: str, counter_speech: str):
        self.hate_speech.append(hate_speech)
        self.counter_speech.append(counter_speech)
