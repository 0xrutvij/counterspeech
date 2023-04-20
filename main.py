from datasets import load_dataset

from hs_datasets import DatasetFactory

conan = DatasetFactory.get_dataset("conan")
mtconan = DatasetFactory.get_dataset("multi_target_conan")
mtknconan = DatasetFactory.get_dataset("multi_target_kn_gr_conan")

conan_trts = load_dataset("csv", data_files=conan)
print(conan_trts)
mtconan_trts = load_dataset("csv", data_files=mtconan)
print(mtconan_trts)
mtknconan_trts = load_dataset("csv", data_files=mtknconan)
print(mtknconan_trts)
