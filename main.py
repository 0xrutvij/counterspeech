import argparse
import random
from pathlib import Path

import sparknlp
from datasets import Dataset, DatasetDict
from pyspark.sql import DataFrame, SparkSession
from transformers import TrainingArguments

from counterspeech import DatasetFactory, HSCSDataset
from counterspeech.config.gpt_confs import get_dialo_gpt_conf
from counterspeech.config.macros import Macros
from counterspeech.models.dialo_gpt import DialoGPT
from counterspeech.modules.hf_gpt2_trainer import GPT2Trainer
from counterspeech.modules.response_generator import ResponseGenerator
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
    choices=["trainer", "examples", "gp2t_trainer"],
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


def _dialo_gpt2_pipeline():
    model = DialoGPT()
    dataset = load_dataset(HSCSDataset.conan)
    training_args = get_dialo_gpt_conf(
        dataset_name=HSCSDataset.conan.name,
        output_base_dir=Macros.result_dir,
        num_train_epochs=10,
        batch_size=32,
        learning_rate=5e-5,
    )

    GPT2Trainer.finetune(
        model=model,
        dataset=dataset,
        training_args=TrainingArguments(**training_args),
    )

    GPT2Trainer.evaluate(
        model=ResponseGenerator(
            training_args["output_dir"],
            decoding_conf={
                "min_new_tokens": 20,
                "max_new_tokens": 100,
                "no_repeat_ngram_size": 5,
                "num_beams": 10,
            },
        ),
        data=dataset,
        prompt=Macros.gpt2_prompt,
        show_n=10,
        batch_size=32,
    )


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
