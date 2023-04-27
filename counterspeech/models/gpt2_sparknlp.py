from dataclasses import dataclass
from typing import Optional

from pyspark.sql import SparkSession
from sparknlp.annotator import DocumentAssembler, GPT2Transformer
from sparknlp.base import Pipeline

from counterspeech.config.macros import Macros
from counterspeech.datasets import HSCSDataset


@dataclass
class GPT2onSpark:
    spark_session: SparkSession
    dataset: HSCSDataset
    model_name: str = "gpt2_csgen"
    output_col_name: str = "generation"
    seed_num: Optional[int] = None
    load_model_path: Optional[str] = None

    def __post_init__(self):
        self.model_name = (
            f"{self.model_name}_{self.seed_num}"
            if self.seed_num is not None
            else self.model_name
        )
        self.model_dir = Macros.result_dir / "model_sparknlp"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.documentAssembler = (
            DocumentAssembler().setInputCol("hate_speech").setOutputCol("documents")
        )
        self.gpt2 = None
        self.batch_size = Macros.BATCH_SIZE
        self.max_output_length = self.dataset.max_output_length
        if self.load_model_path is None:
            self.gpt2 = (
                GPT2Transformer.pretrained("gpt2")
                .setTask("generation")
                .setInputCols(["documents"])
                .setMaxOutputLength(self.max_output_length)
                .setOutputCol(self.output_col_name)
                .setBatchSize(self.batch_size)
            )
        else:
            self.gpt2 = (
                GPT2Transformer.loadSavedModel(self.load_model_path, self.spark_session)
                .setInputCols(["documents"])
                .setMaxOutputLength(self.max_output_length)
                .setOutputCol(self.output_col_name)
            )
        # end if
        self.pipeline = Pipeline(stages=[self.documentAssembler, self.gpt2])

    def train(self, train_df):
        print(f"Training Dataset Count: {train_df.count()}")
        self.pipeline = self.pipeline.fit(train_df)
        self.pipeline.stages[-1].write().overwrite().save(
            str(self.model_dir / self.model_name)
        )
        return

    def predict(self, test_df):
        pred_df = (
            self.pipeline.transform(test_df)
            .select(
                "hate_speech",
                "counter_speech",
                self.output_col_name,
            )
            .toPandas()
        )
        pred_df[self.output_col_name] = pred_df[self.output_col_name].apply(
            lambda x: x[0]["result"].replace("\n", "<nl>")
        )
        pred_df.to_csv(
            str(self.model_dir / self.model_name / "test_results.csv"), sep=","
        )
        return pred_df