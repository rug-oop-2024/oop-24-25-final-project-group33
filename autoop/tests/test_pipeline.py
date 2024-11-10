from sklearn.datasets import fetch_openml
import unittest
import pandas as pd

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.regression.regression import MultipleLinearRegression
from autoop.core.ml.metric import MeanSquaredError


class TestPipeline(unittest.TestCase):
    """
    TestPipeline class for testing the pipeline functionality.
    Methods:
        setUp(): Set up the test environment.
        test_init(): Test the initialization of the Pipeline instance.
        test_preprocess_features(): Test the _preprocess_features method.
        test_split_data(): Test data splitting into training and testing sets.
        test_train(): Test the training process of the pipeline.
        test_evaluate(): Test the evaluation process of the pipeline.
    """

    def setUp(self) -> None:
        """
        Set up the test environment.
        Initializes dataset, features, pipeline, and dataset size.
        Attributes:
            dataset (Dataset): The dataset object created from the dataframe.
            features (list): List of detected feature types.
            pipeline (Pipeline): The pipeline object for the model.
            ds_size (int): The size of the dataset.
        """

        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=MultipleLinearRegression(),
            input_features=list(
                filter(lambda x: x.name != "age", self.features)
                ),
            target_f=Feature(name="age", type="numerical"),
            metrics=[MeanSquaredError()],
            split=0.8
        )
        self.ds_size = data.data.shape[0]

    def test_init(self):
        """
        Test the initialization of the Pipeline instance.
        """

        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self):
        """
        Test the _preprocess_features method of the pipeline.
        Args:
            self: The test instance.
        Returns:
            None
        """
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self):
        """
        Test data splitting into training and testing sets.
        Args:
            self: The test instance.
        Returns:
            None
        """

        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.assertEqual(
            self.pipeline._train_X[0].shape[0], int(0.8 * self.ds_size)
            )
        self.assertEqual(
            self.pipeline._test_X[0].shape[0],
            self.ds_size - int(0.8 * self.ds_size)
            )

    def test_train(self):
        """
        Test the training process of the pipeline.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.parameters)

    def test_evaluate(self):
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.pipeline._evaluate()
        self.assertIsNotNone(self.pipeline._predictions)
        self.assertIsNotNone(self.pipeline._metrics_results)
        self.assertEqual(len(self.pipeline._metrics_results), 1)
