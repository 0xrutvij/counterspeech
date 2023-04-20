import requests
from pathlib import Path
from dataclasses import asdict, dataclass, field

import pandas as pd

URLS = {
    "conan": (
        "https://raw.githubusercontent.com/marcoguerini/CONAN/master/CONAN/CONAN.csv"
    ),
    "multi_target_conan": (
        "https://raw.githubusercontent.com/marcoguerini/"
        "CONAN/master/Multitarget-CONAN/Multitarget-CONAN.csv"
    ),
    "multi_target_kn_gr_conan": (
        "https://raw.githubusercontent.com/marcoguerini/"
        "CONAN/master/multitarget_KN_grounded_CN/multitarget_KN_grounded_CN.csv"
    ),
}


def _download_file(url: str, path: Path) -> Path:
    """Download a file from a URL to a given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return path


def download_data_csv(dataset: str, data_dir: Path) -> Path:
    """Download a dataset from a URL to a given path."""
    url = URLS.get(dataset)
    if url is None:
        raise ValueError(f"Dataset {dataset} is not supported")
    path = data_dir / f"{dataset}.csv"
    if path.exists():
        return path
    return _download_file(url, path)


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

    def _preprocess(self) -> Pairs:
        match self.dataset_name:
            case "conan":
                return self._preprocess_conan()

            case "multi_target_conan":
                return self._preprocess_multi_target_conan()

            case "multi_target_kn_gr_conan":
                return self._preprocess_multi_target_kn_gr_conan()

            case _:
                raise NotImplementedError(
                    f"Dataset {self.dataset_name} is not supported"
                )

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


class DatasetFactory:
    data_dir = Path("data")

    @classmethod
    def get_dataset(cls, dataset: str) -> dict[str, str]:
        data_csv = download_data_csv(dataset, cls.data_dir)
        preprocessor = PreProcessor(data_csv)

        if not preprocessor.train_csv.exists() or not preprocessor.test_csv.exists():
            preprocessor.run()

        return {
            "train": str(preprocessor.train_csv),
            "test": str(preprocessor.test_csv),
        }
