import os
import tempfile
import unittest
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    FEATURES_COLS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    TARGET_COL = [
        "delay"
    ]


    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        # self.data = pd.read_csv(filepath_or_buffer="../data/data.csv")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, 'data', 'data.csv')
        self.data = pd.read_csv(
            filepath_or_buffer=data_path.replace('tests/model/', ''),
            low_memory=False
        )

    def test_model_preprocess_for_training(
            self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)

    def test_model_preprocess_for_serving(
            self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

    def test_model_fit(
            self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        _, features_validation, _, target_validation = train_test_split(features, target, test_size=0.33,
                                                                        random_state=42)

        self.model.fit(
            features=features,
            target=target
        )

        predicted_target = self.model._model.predict(
            features_validation
        )

        report = classification_report(target_validation, predicted_target, output_dict=True)
        print(report)
        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30

    def test_model_predict(
            self
    ):
        features, target = self.model.preprocess(data=self.data, target_column="delay")

        # Ensure the model is trained before prediction
        self.model.fit(features=features, target=target)

        predicted_targets = self.model.predict(
            features=features
        )

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)

    def test_predict_before_training(self):
        features = self.model.preprocess(data=self.data)

        # Try predicting without training the model
        with self.assertRaises(ValueError, msg="Model hasn't been trained yet!"):
            self.model.predict(features=features)

    def test_save_and_load_model(self):
        features, target = self.model.preprocess(data=self.data, target_column="delay")

        # Train the model
        self.model.fit(features=features, target=target)

        # Predict before saving and loading
        predictions_before = self.model.predict(features=features)

        # Use a temporary file to save and load the model
        with tempfile.NamedTemporaryFile(suffix=".joblib") as temp_file:
            # Save the model
            self.model.save_model(filepath=temp_file.name)

            # Load the model
            self.model.load_model(filepath=temp_file.name)

        # Predict after loading
        predictions_after = self.model.predict(features=features)

        # Ensure the predictions remain consistent
        assert predictions_before == predictions_after, "Predictions mismatch before and after loading the model"

    def test_save_without_training(self):
        # Try saving the model without training
        with tempfile.NamedTemporaryFile(suffix=".joblib") as temp_file:
            with self.assertRaises(ValueError, msg="No model has been trained yet. Cannot save."):
                self.model.save_model(filepath=temp_file.name)

    def test_preprocess_with_all_necessary_columns(self):
        # Use a subset of the data with the necessary columns
        necessary_data = self.data[["OPERA", "TIPOVUELO", "MES"]].copy()

        # This should not raise any exception
        features = self.model.preprocess(data=necessary_data)

        assert isinstance(features, pd.DataFrame)

    def test_preprocess_missing_necessary_column(self):
        necessary_columns = ["OPERA", "TIPOVUELO", "MES"]

        for col in necessary_columns:
            # Drop one necessary column at a time
            data_missing_col = self.data.drop(columns=[col])

            with self.assertRaises(ValueError, msg=f"Missing necessary column: {col}"):
                self.model.preprocess(data=data_missing_col)

    def test_ensure_top_features_in_dataframe(self):
        # Create a mock dataframe that's missing some of the top features
        mock_data = pd.DataFrame({
            'OPERA': ['Some Value'],
            'TIPOVUELO': ['Some Value'],
            'MES': [1]
        })

        # Use the preprocess method to get features
        features = self.model.preprocess(data=mock_data)

        # Check if all top features are present in the features dataframe
        for col in self.model._top_features:
            self.assertIn(col, features.columns)

        # Check if the added columns have the default value of 0
        for col in self.model._top_features:
            if col not in mock_data.columns:
                self.assertTrue((features[col] == 0).all())

    def test_is_high_season(self):
        # High season dates
        self.assertEqual(DelayModel._is_high_season("2023-12-16 12:00:00"), 1)
        self.assertEqual(DelayModel._is_high_season("2023-01-02 12:00:00"), 1)
        self.assertEqual(DelayModel._is_high_season("2023-07-16 12:00:00"), 1)
        self.assertEqual(DelayModel._is_high_season("2023-09-12 12:00:00"), 1)

        # Non-high season dates
        self.assertEqual(DelayModel._is_high_season("2023-04-16 12:00:00"), 0)
        self.assertEqual(DelayModel._is_high_season("2023-05-02 12:00:00"), 0)

    def test_get_period_day(self):
        # Testing different periods of the day
        self.assertEqual(DelayModel._get_period_day("2023-12-16 06:00:00"),
                         "mañana")
        self.assertEqual(DelayModel._get_period_day("2023-12-16 14:00:00"),
                         "tarde")
        self.assertEqual(DelayModel._get_period_day("2023-12-16 20:00:00"),
                         "noche")
        self.assertEqual(DelayModel._get_period_day("2023-12-16 02:00:00"),
                         "noche")

    def test_get_min_diff(self):
        # Mock data for testing
        data = {
            "Fecha-O": "2023-12-16 14:30:00",
            "Fecha-I": "2023-12-16 14:00:00"
        }
        self.assertEqual(DelayModel._get_min_diff(data), 30.0)

        data = {
            "Fecha-O": "2023-12-16 14:00:00",
            "Fecha-I": "2023-12-16 14:30:00"
        }
        self.assertEqual(DelayModel._get_min_diff(data), -30.0)


