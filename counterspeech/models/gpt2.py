import os
import random
from typing import Optional

import numpy as np
import torch
from torch.backends import cudnn
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

from counterspeech.config.macros import Macros


class GPT2:
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if model_name_or_path is None:
            model_name_or_path = Macros.models[self.__class__.__name__.lower()]

        if seed is not None:
            self._seed_everything(seed)

        self.device = device
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name_or_path)
        self.model = OpenAIGPTLMHeadModel.from_pretrained(model_name_or_path)
        self.prompt = Macros.gpt2_prompt

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(device)
        self.model.eval()

    def _seed_everything(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    def generate(self, input_seq: str, max_length: int):
        input_with_prompt = (
            f"{self.prompt} offensive post: {input_seq} -> counterspeech: "
        )
        input_ids = self.tokenizer.encode(input_with_prompt, return_tensors="pt")
        outputs = self.model.generate(
            input_ids.to(self.device), max_length=max_length, do_sample=True
        )
        for i, beam in enumerate(outputs):
            out_texts = self.tokenizer.decode(beam, skip_special_tokens=True)
            print(f"{i}: {out_texts}\n")

        return outputs
