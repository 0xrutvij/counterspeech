import random
from collections import defaultdict

import torch
from colorama import Fore, Style
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments

from counterspeech.metrics import BLEU, Toxicity
from counterspeech.models import GPT2, DialoGPT
from counterspeech.models.utils import tokenize
from counterspeech.modules.response_generator import ResponseGenerator


class GPT2Trainer:
    bleu = BLEU()
    tox = Toxicity()

    @classmethod
    def finetune(
        cls,
        dataset: DatasetDict,
        model: DialoGPT | GPT2,
        training_args: TrainingArguments,
        sched: torch.optim.lr_scheduler._LRScheduler,
        optim: torch.optim.Optimizer,
        freeze_n: int = 0,
    ):
        """Fine Tune the GPT2 models."""

        print(f"Fine Tuning {model.model_name_or_path} with {freeze_n} layers frozen")

        model.model = model.freeze_n_layers(model.model, freeze_n)

        tokenized_dataset = tokenize(dataset, model.tokenizer)

        trainer = Trainer(
            model=model.model,
            args=training_args,
            data_collator=model.data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"],
            tokenizer=model.tokenizer,
            optimizers=(optim, sched),
        )

        print("Training...")
        trainer.train()

        print("Evaluating...")
        trainer.evaluate()

        print("Saving model...")
        trainer.save_model()

    @staticmethod
    def add_prompt(prompt: str, data: Dataset):
        """Add a prompt to the beginning of each example in the dataset."""

        hs_refs_map = defaultdict(list)
        hs_w_prompt_map = dict()

        for example in data:
            hate_speech = example["hate_speech"]
            counter_speech = example["counter_speech"]

            hs_w_prompt = f"{prompt}\noffensive post: {hate_speech}\ncounterspeech: "

            if hate_speech not in hs_w_prompt_map:
                hs_w_prompt_map[hate_speech] = hs_w_prompt
            hs_refs_map[hate_speech].append(counter_speech)

        hate_speech = list(hs_w_prompt_map.values())
        counter_speech_refs = list(hs_refs_map.values())

        return hate_speech, counter_speech_refs

    @classmethod
    def show_comp(
        cls,
        hate_speech: list[str],
        preds: list[str],
        counter_speech_refs: list[list[str]],
        n: int,
    ):
        for i, (hs, pred, exp) in enumerate(
            zip(hate_speech, preds, counter_speech_refs)
        ):
            if i >= n:
                break
            print(
                (
                    Fore.RED
                    + hs
                    + "\n"
                    + Style.RESET_ALL
                    + "model >> "
                    + Fore.BLUE
                    + pred
                    + "\n"
                    + Style.RESET_ALL
                    + "gold >> "
                    + Fore.GREEN
                    + random.choice(exp)
                    + "\n"
                    + Style.RESET_ALL
                    + "\n"
                )
            )

    @classmethod
    def _generate_counter_speech(
        cls,
        model: ResponseGenerator,
        prompt: str,
        data: Dataset,
        batch_size: int = 32,
    ):
        hate_speech: list[str]
        counter_speech_refs: list[list[str]]

        hate_speech, counter_speech_refs = cls.add_prompt(prompt, data)

        preds = model.generate_responses(hate_speech, batch_size)
        return hate_speech, preds, counter_speech_refs

    @classmethod
    def _evaluate_bleu(
        cls,
        preds: list[str],
        counter_speech_refs: list[list[str]],
    ):
        bleu = cls.bleu.calculate_score(preds, counter_speech_refs)
        print(f"BLEU: {bleu}")

    @classmethod
    def _evaluate_toxicity(
        cls,
        preds: list[str],
    ):
        toxicity = cls.tox.calculate_score(preds)
        print(
            f"Toxicity:"
            f"\n\tAvg: {toxicity:.3f}"
            f"\n\tMax: {cls.tox.max_toxicity:.3f}"
            f"\n\tTox Ratio: {cls.tox.toxicity_ratio:.3f}"
        )

    @classmethod
    def evaluate(
        cls,
        model: ResponseGenerator,
        prompt: str,
        data: DatasetDict,
        show_n: int = 5,
        batch_size: int = 32,
    ):
        """Evaluate the model on the dataset."""

        for split in data:
            print(f"Evaluating on {split}...")
            hate_speech, preds, counter_speech_refs = cls._generate_counter_speech(
                model, prompt, data[split], batch_size
            )

            print(f"Evaluating BLEU on {split}...")
            cls._evaluate_bleu(preds, counter_speech_refs)

            print(f"Evaluating Toxicity on {split}...")
            cls._evaluate_toxicity(preds)

            if show_n > 0:
                print(f"Showing {show_n} examples from {split}...")
                cls.show_comp(hate_speech, preds, counter_speech_refs, show_n)

            print("-" * 50)
