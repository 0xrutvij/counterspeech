from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

from .utils import DataCollatorForLanguageModeling, tokenize


def freeze_n_layers(model: GPT2LMHeadModel, n: int):
    """Freeze the first n layers of the model."""
    for i in range(n):
        for param in model.h[i].parameters():
            param.requires_grad = False
    return model


def finetune_dialo_gpt(
    dataset: DatasetDict,
    training_args: TrainingArguments,
    model_name: str = "microsoft/DialoGPT-medium",
    freeze_n: int = 0,
):
    """Train the DialoGPT model."""

    print(f"Fine Tuning {model_name} with {freeze_n} layers frozen")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = freeze_n_layers(model, freeze_n)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tokenized_dataset = tokenize(dataset, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
    )

    print("Training...")
    trainer.train()

    print("Evaluating...")
    trainer.evaluate()

    print("Saving model...")
    trainer.save_model()
