from dataclasses import dataclass
from typing import Optional

from pyspark.sql import SparkSession
from sparknlp.annotator import DocumentAssembler, T5Transformer
from sparknlp.base import Pipeline

from counterspeech.config.macros import Macros
from counterspeech.datasets import HSCSDataset


@dataclass
class T5onSpark:
    spark_session: SparkSession
    dataset: HSCSDataset
    model_name: str = "t5_csgen"
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
        self.t5 = None
        self.batch_size = Macros.BATCH_SIZE
        if self.load_model_path is None:
            self.t5 = (
                T5Transformer.pretrained("google_t5_small_ssm_nq")
                .setInputCols(["documents"])
                .setMaxOutputLength(self.dataset.max_output_length)
                .setOutputCol(self.output_col_name)
                .setBatchSize(self.batch_size)
            )
        else:
            self.t5 = (
                T5Transformer.loadSavedModel(self.load_model_path, self.spark_session)
                .setInputCols(["documents"])
                .setMaxOutputLength(self.dataset.max_output_length)
                .setOutputCol(self.output_col_name)
            )
        self.pipeline = Pipeline(stages=[self.documentAssembler, self.t5])

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
            .select("hate_speech", "counter_speech", "generation")
            .toPandas()
        )

        pred_df[self.output_col_name] = pred_df[self.output_col_name].apply(
            lambda x: x[0]["result"].replace("\n", "<nl>")
        )

        file_name = (
            "test_results.csv"
            if self.seed_num is None
            else f"test_results_{self.seed_num}.csv"
        )

        pred_df.to_csv(str(self.model_dir / self.model_name / file_name), sep=",")
        return pred_df
