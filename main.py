import argparse
import random
from pathlib import Path

import sparknlp
from datasets import Dataset, DatasetDict
from pyspark.sql import DataFrame, SparkSession

from counterspeech import DatasetFactory, HSCSDataset
from counterspeech.config.macros import Macros
from counterspeech.modules.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="conan",
    choices=["conan", "multi_target_conan", "multi_target_kn_gr_conan"],
    help="dataset to use",
)
parser.add_argument(
    "--model",
    type=str,
    default="gpt2",
    choices=["gpt2", "t5", "bart"],
    help="model to use",
)
parser.add_argument(
    "--run",
    type=str,
    default="trainer",
    choices=["trainer", "examples"],
    help="task to be run",
)
parser.add_argument(
    "--seed_num",
    type=int,
    default=None,
    help="seed for train/test split",
)


def set_seed(seed_num: int):
    rand_seed = Macros.RAND_SEED if seed_num is None else Macros.RAND_SEED + seed_num
    random.seed(rand_seed)


def load_dataset(dataset: HSCSDataset) -> DatasetDict:
    return DatasetFactory.get_dataset(dataset, Macros.dataset_dir)


def huggingface_dataset_to_spark_df(ss: SparkSession, hf_dataset: Dataset) -> DataFrame:
    ss.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return ss.createDataFrame(hf_dataset)


def run_trainer(args: argparse.Namespace):
    Macros.set_storage_dir(Path("/nas1-nfs1/data/jxl115330/csgen4hs"))
    trainer = Trainer(
        model_type=args.model,
        dataset_name=args.dataset,
        seed_num=args.seed_num,
    )
    trainer.train()
    trainer.evaluate()


def examples(args: argparse.Namespace):
    spark = sparknlp.start()
    ddict = load_dataset(HSCSDataset[args.dataset])
    print(ddict)
    tr_df = huggingface_dataset_to_spark_df(spark, ddict["train"])
    tr_df.show()


FUNC_MAP = {
    "trainer": run_trainer,
    "examples": examples,
}


if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed_num)
    FUNC_MAP[args.run](args)
