from pathlib import Path

import pandas as pd

from .pairs import Pairs


class PreProcessor:
    def __init__(self, data_csv: Path):
        self.data_csv = data_csv
        self.dataset_name = data_csv.stem
        self.train_csv = data_csv.parent / f"{data_csv.stem}_train.csv"
        self.test_csv = data_csv.parent / f"{data_csv.stem}_test.csv"

    def run(self):
        pairs = self._preprocess()
        pairs_df = pairs.to_pandas()
        train_df = pairs_df.sample(frac=0.8, random_state=42)
        test_df = pairs_df.drop(train_df.index)
        train_df.to_csv(self.train_csv, index=False)
        test_df.to_csv(self.test_csv, index=False)

    @property
    def _preprocessors(self):
        return {
            "conan": self._preprocess_conan,
            "multi_target_conan": self._preprocess_multi_target_conan,
            "multi_target_kn_gr_conan": self._preprocess_multi_target_kn_gr_conan,
        }

    def _preprocess(self) -> Pairs:
        preprocessor = self._preprocessors.get(self.dataset_name)

        if preprocessor is None:
            raise NotImplementedError(f"Dataset {self.dataset_name} is not supported")

        return preprocessor()

    def _preprocess_multi_target_conan(self) -> Pairs:
        pairs = Pairs()
        conan = pd.read_csv(self.data_csv)
        for _, row in conan.iterrows():
            pairs.append(row["HATE_SPEECH"], row["COUNTER_NARRATIVE"])
        return pairs

    def _preprocess_multi_target_kn_gr_conan(self) -> Pairs:
        pairs = Pairs()
        conan = pd.read_csv(self.data_csv)
        for _, row in conan.iterrows():
            pairs.append(row["hate_speech"], row["counter_narrative"])
        return pairs

    def _preprocess_conan(self) -> Pairs:
        pairs = Pairs()
        conan = pd.read_csv(self.data_csv)

        def _is_english(cn_id: str):
            cn_id = cn_id.lower()
            return cn_id.startswith("en") or cn_id.endswith("t1")

        for _, row in conan.iterrows():
            if _is_english(row["cn_id"]):
                pairs.append(row["hateSpeech"], row["counterSpeech"])
        return pairs
