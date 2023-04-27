from dataclasses import dataclass
from typing import Optional

import sparknlp

from counterspeech.config.macros import Macros
from counterspeech.datasets import DatasetFactory, HSCSDataset
from counterspeech.models import BARTonSpark, GPT2onSpark, T5onSpark


@dataclass
class Trainer:
    model_type: str
    dataset_name: str
    seed_num: Optional[int] = None
    use_cuda: bool = True

    def __post_init__(self):
        spark = sparknlp.start(gpu=self.use_cuda)
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

        if self.model_type == "gpt2":
            self.model_class = GPT2onSpark
        elif self.model_type == "t5":
            self.model_class = T5onSpark
        elif self.model_type == "bart":
            self.model_class = BARTonSpark
        else:
            raise ValueError(
                f"Invalid model type: {self.model_type}. Must be one of: gpt2, t5, bart"
            )

        self.model_obj = self.model_class(
            spark_session=self.spark_session, seed_num=self.seed_num
        )

        if self.dataset_name == "conan":
            dataset = HSCSDataset.conan
        elif self.dataset_name == "multi_target_conan":
            dataset = HSCSDataset.multi_target_conan
        elif self.dataset_name == "multi_target_kn_gr_conan":
            dataset = HSCSDataset.multi_target_kn_gr_conan
        else:
            raise ValueError(
                f"Invalid dataset name: {self.dataset_name}. "
                "Must be one of: conan, multi_target_conan, multi_target_kn_gr_conan"
            )

        hf_dataset_dict = DatasetFactory.get_dataset(dataset, Macros.dataset_dir)
        self.train_df = spark.createDataFrame(hf_dataset_dict["train"])
        self.test_df = spark.createDataFrame(hf_dataset_dict["test"])
        self.spark_session = spark

    def train(self):
        self.model_obj.train(self.train_df)
        return

    def evaluate(self):
        pred_df = self.model_obj.predict(self.test_df)
        # df['result'] = df['result'].apply(lambda x: x[0])
        # cls_report = classification_report(df.class_index, df.result)
        # acc_score = accuracy_score(df.class_index, df.result)
        return pred_df
