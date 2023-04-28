import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from datasets import DatasetDict
from torch.backends import cudnn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

from counterspeech.config.macros import Macros
from counterspeech.models.utils import DataCollatorForLanguageModeling, tokenize


@dataclass
class GPT2:
    """GPT2 model for text generation."""

    model_name_or_path: Optional[str] = "gpt2"
    local_model_path: Optional[str] = None
    seed: Optional[int] = None
    device: Optional[str] = None

    def __post_init__(self):
        if self.seed is not None:
            self._seed_everything(self.seed)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_name_or_path is None:
            self.model_name_or_path = Macros.models[self.__class__.__name__.lower()]

        self.model_dir = (
            Macros.result_dir / "model_hf" / self.model_name.replace("/", "_")
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.prompt = Macros.gpt2_prompt
        self.config = GPT2Config.from_pretrained(
            self.model_name, output_hidden_states=False
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name_or_path, config=self.config
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()

    def _seed_everything(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    def freeze_n_layers(self, model: GPT2LMHeadModel, n: int):
        """Freeze the first n layers of the model."""
        for i in range(n):
            for param in model.h[i].parameters():
                param.requires_grad = False
        return model

    def finetune_gpt2(
        self,
        dataset: DatasetDict,
        training_args: TrainingArguments,
        freeze_n: int = 0,
    ):
        self.model = self.freeze_n_layers(self.model, freeze_n)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        tokenized_dataset = tokenize(dataset, self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"],
            tokenizer=self.tokenizer,
        )
        print("Training...")
        trainer.train()

        print("Evaluating...")
        trainer.evaluate()

        print("Saving model...")
        trainer.save_model()
        return

    def generate(self, input_seq: str, max_length: int):
        eos_token = self.tokenizer.eos_token
        input_with_prompt = f"{self.prompt}: {input_seq} {eos_token}"
        input_ids = self.tokenizer.encode(input_with_prompt, return_tensors="pt")
        outputs = self.model.generate(
            input_ids.to(self.device), max_length=max_length, do_sample=True
        )
        for i, beam in enumerate(outputs):
            out_texts = self.tokenizer.decode(beam, skip_special_tokens=True)
            print(f"{i}: {out_texts}\n")
        # end for
        return outputs
