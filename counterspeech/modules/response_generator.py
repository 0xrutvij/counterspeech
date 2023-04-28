from typing import Any

import torch
import tqdm
from colorama import Fore, Style
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
)


class ResponseGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        decoding_conf: dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.decoding_conf = decoding_conf
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        self.model = self.model.to(self.device)
        torch.manual_seed(self.seed)

        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_responses(
        self, input_labels: list[str], batch_size: int = 1
    ) -> list[str]:
        responses = []
        for i in tqdm.tqdm(range(0, len(input_labels), batch_size)):
            batch_end = i + batch_size
            batch = input_labels[i:batch_end]
            batch_responses = self.generate_responses_batch(batch)
            responses.extend(batch_responses)
        return responses

    def generate_responses_batch(self, input_labels: list[str]) -> list[str]:
        input_labels = [
            input_text + self.tokenizer.eos_token for input_text in input_labels
        ]

        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer(
            input_labels,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        input_len = tokenized_inputs.input_ids.shape[1]

        params_for_generation = self._params_for_generation(input_len)

        output_ids = self.model.generate(
            **tokenized_inputs,
            **params_for_generation,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response_ids = output_ids[:, input_len:].tolist()
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        return responses

    def _params_for_generation(self, input_len: int) -> dict[str, Any]:
        params_for_generation = self.decoding_conf.copy()
        lps = []

        if (
            "min_new_tokens" in params_for_generation
            and params_for_generation["min_new_tokens"] is not None
        ):
            min_new_tokens = params_for_generation["min_new_tokens"]
            min_new_token_lp = MinNewTokensLengthLogitsProcessor(
                min_new_tokens=min_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            lps.append(min_new_token_lp)
            min_length_lp = MinLengthLogitsProcessor(
                min_length=input_len + min_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            lps.append(min_length_lp)
            params_for_generation.pop("min_new_tokens")
        else:
            min_length_lp = MinLengthLogitsProcessor(
                min_length=input_len,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            lps.append(min_length_lp)

        params_for_generation["logits_processors"] = LogitsProcessorList(lps)

        return params_for_generation

    def respond(self, input_text: str) -> str:
        return self.generate_responses([input_text])[0]

    def interact(self):
        prompt = Fore.RED + "Hate Speech: " + Style.RESET_ALL
        input_text = input(prompt)
        while input_text != "":
            response = self.respond(input_text)
            print(Fore.GREEN + "Response: " + Style.RESET_ALL + response)
            input_text = input(prompt)
