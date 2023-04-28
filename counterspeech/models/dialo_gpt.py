import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.backends import cudnn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel

from counterspeech.config.macros import Macros

from .utils import DataCollatorForLanguageModeling


@dataclass
class DialoGPT:
    model_name_or_path: Optional[str] = "microsoft/DialoGPT-medium"
    seed: Optional[int] = None
    device: Optional[str] = None

    def __post_init__(self):
        if self.seed is not None:
            self._seed_everything(self.seed)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_name_or_path is None:
            self.model_name_or_path = Macros.models[self.__class__.__name__.lower()]

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
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
