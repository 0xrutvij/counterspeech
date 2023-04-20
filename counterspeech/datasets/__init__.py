from collections import namedtuple
from enum import Enum
from pathlib import Path

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from .preprocesssor import PreProcessor
from .utils import download_file

DatasetProperties = namedtuple("DatasetProperties", "url")

GH_SOURCE1 = "https://raw.githubusercontent.com/marcoguerini/CONAN/master"


class HSCSDataset(DatasetProperties, Enum):
    conan = DatasetProperties(
        f"{GH_SOURCE1}/CONAN/CONAN.csv",
    )
    multi_target_conan = DatasetProperties(
        f"{GH_SOURCE1}/Multitarget-CONAN/Multitarget-CONAN.csv",
    )
    multi_target_kn_gr_conan = DatasetProperties(
        f"{GH_SOURCE1}/multitarget_KN_grounded_CN/multitarget_KN_grounded_CN.csv",
    )

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
        preprocessor = PreProcessor(data_csv)

        if not preprocessor.train_csv.exists() or not preprocessor.test_csv.exists():
            preprocessor.run()

        data_files = {
            "train": str(preprocessor.train_csv),
            "test": str(preprocessor.test_csv),
        }

        dsdc = load_dataset("csv", data_files=data_files)
        assert isinstance(dsdc, DatasetDict)

        return dsdc
