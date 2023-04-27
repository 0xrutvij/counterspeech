from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from .preprocesssor import ConanPreprocessor
from .utils import download_file

GH_SOURCE1 = "https://raw.githubusercontent.com/marcoguerini/CONAN/master"


@dataclass
class DatasetProperties:
    url: str
    max_output_length: int
    train_ratio: float


class HSCSDataset(DatasetProperties, Enum):
    conan = DatasetProperties(
        url=f"{GH_SOURCE1}/CONAN/CONAN.csv",
        max_output_length=150,
        train_ratio=0.8,
    )
    multi_target_conan = DatasetProperties(
        url=f"{GH_SOURCE1}/Multitarget-CONAN/Multitarget-CONAN.csv",
        max_output_length=150,
        train_ratio=0.8,
    )
    multi_target_kn_gr_conan = DatasetProperties(
        url=f"{GH_SOURCE1}/multitarget_KN_grounded_CN/multitarget_KN_grounded_CN.csv",
        max_output_length=150,
        train_ratio=0.8,
    )

    def __init__(self, dataset_properties: DatasetProperties):
        self.url = dataset_properties.url
        self.max_output_length = dataset_properties.max_output_length
        self.train_ratio = dataset_properties.train_ratio

    def __str__(self) -> str:
        return self.name

    def download_path(self, data_dir: Path) -> Path:
        return data_dir / f"{self}.csv"

    def download(self, data_dir: Path) -> Path:
        """Download a dataset from a URL to a given path."""
        path = self.download_path(data_dir)
        if path.exists():
            return path
        return download_file(self.url, path)


class DatasetFactory:
    @classmethod
    def get_dataset(cls, dataset: HSCSDataset, data_dir: Path) -> DatasetDict:
        data_csv = dataset.download(data_dir)
        preprocessor = ConanPreprocessor(data_csv)

        if not preprocessor.train_csv.exists() or not preprocessor.test_csv.exists():
            preprocessor.run()

        data_files = {
            "train": str(preprocessor.train_csv),
            "test": str(preprocessor.test_csv),
            "val": str(preprocessor.val_csv),
        }

        dsdc = load_dataset("csv", data_files=data_files)
        assert isinstance(dsdc, DatasetDict)

        return dsdc
