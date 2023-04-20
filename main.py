from functools import partial
from pathlib import Path

from counterspeech import DatasetFactory, HSCSDataset

load_dataset_at_path = partial(DatasetFactory.get_dataset, data_dir=Path("data"))


conan_trts = load_dataset_at_path(HSCSDataset.conan)
print(conan_trts)

mtconan_trts = load_dataset_at_path(HSCSDataset.multi_target_conan)
print(mtconan_trts)

mtknconan_trts = load_dataset_at_path(HSCSDataset.multi_target_kn_gr_conan)
print(mtknconan_trts)
