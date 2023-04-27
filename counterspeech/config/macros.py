from pathlib import Path


class Macros:
    storage_dir: Path = Path(__file__).parent.parent.parent
    result_dir: Path = storage_dir / "_results"
    download_dir: Path = storage_dir / "_downloads"
    dataset_dir: Path = download_dir / "datasets"
    log_dir: Path = storage_dir / "_logs"
    info = [
        __package__,
        __name__,
        __file__,
    ]

    @classmethod
    def set_storage_dir(cls, storage_dir: Path):
        cls.storage_dir = storage_dir
        cls.result_dir = storage_dir / "_results"
        cls.download_dir = storage_dir / "_downloads"
        cls.dataset_dir = cls.download_dir / "datasets"
        cls.log_dir = storage_dir / "_logs"

    FMT_INT = "{:,d}"
    FMT_PER = "{:.1%}"
    FMT_FLOAT = "{:,.5f}"
    RAND_SEED = 27

    models = {"gpt2": "openai-gpt"}

    # ==========
    # sparknlp
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 5

    # ==========
    # conan dataset
    conan_data_dir = dataset_dir / "conan"
    conan_csv_file = conan_data_dir / "CONAN.csv"
    conan_max_output_length = 150
    train_ratio = 0.8

    # ==========
    # prompt
    gpt2_prompt = "Generate counterspeech to the given offensive post."

    # ==========
    # dataset
    hatexplain_data_dir = dataset_dir / "HateXplain" / "Data"
    hatexplain_data_file = hatexplain_data_dir / "dataset.json"
    hatexplain_labels_two_classes = {
        "toxic": ["hate", "offensive"],
        "non-toxic": ["normal"],
    }
