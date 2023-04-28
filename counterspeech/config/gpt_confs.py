from typing import Optional


def get_dialo_gpt_conf(
    dataset_name: str,
    output_base_dir: str,
    num_train_epochs: int = 3,
    batch_size: int = 32,
    report_to: Optional[str] = None,
    learning_rate: float = 5e-5,
):
    return {
        "output_dir": f"{output_base_dir}/dialo-gpt-{dataset_name}",
        "report_to": report_to,
        "evaluation_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "save_total_limit": 1,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "metric_for_best_model": "loss",
    }


def get_gpt2_conf(
    dataset_name: str,
    output_base_dir: str,
    num_train_epochs: int = 3,
    batch_size: int = 32,
    report_to: Optional[str] = None,
):
    return {
        "output_dir": f"{output_base_dir}/gpt2-{dataset_name}",
        "report_to": report_to,
        "evaluation_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "save_total_limit": 1,
        "learning_rate": 5e-5,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "metric_for_best_model": "loss",
    }
