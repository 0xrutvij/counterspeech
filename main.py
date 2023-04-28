import argparse
import random

from datasets import DatasetDict
from transformers import TrainingArguments

from counterspeech import DatasetFactory, HSCSDataset
from counterspeech.config.gpt_confs import get_dialo_gpt_conf
from counterspeech.config.macros import Macros
from counterspeech.models.dialo_gpt import DialoGPT
from counterspeech.modules.hf_gpt2_trainer import GPT2Trainer
from counterspeech.modules.response_generator import ResponseGenerator

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
    default="examples",
    choices=["examples", "gp2t_trainer"],
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
    ddict = load_dataset(HSCSDataset[args.dataset])
    print(ddict)


FUNC_MAP = {
    "examples": examples,
}


if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed_num)
    FUNC_MAP[args.run](args)
