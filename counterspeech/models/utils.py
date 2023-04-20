from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


def tokenize(dataset: DatasetDict, tokenizer: AutoTokenizer):
    """Tokenize the dataset."""

    def tokenize_function(row):
        EOS = tokenizer.eos_token
        return tokenizer(f"{row['hate_speech']}{EOS}{row['counter_speech']}{EOS}")

    return dataset.map(
        tokenize_function,
        remove_columns=["hate_speech", "counter_speech"],
    )


class DCForDialog(DataCollatorForLanguageModeling):
    """To train the language model only on the response by masking the context."""

    def __call__(self, batch):
        batch = super().__call__(batch)
        labels = batch["labels"]

        for i in range(len(labels)):
            eos_eoc = labels[i].tolist().index(self.tokenizer.eos_token_id)
            labels[i, : eos_eoc + 1] = -100

        return batch
