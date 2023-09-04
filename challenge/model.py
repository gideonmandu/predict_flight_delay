import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from joblib import dump, load
from typing import Tuple, Union, List

# from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DelayModel:
    def __init__(self, scale=4.4402380952380955, threshold_in_minutes=15):
        self._model = None  # Model should be saved in this attribute.
        self._top_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]
        self._threshold_in_minutes = threshold_in_minutes
        self._scale = scale

    @staticmethod
    def _is_high_season(fecha):
        fecha_año = int(fecha.split("-")[0])
        fecha = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
        range1_min = datetime.strptime("15-Dec", "%d-%b").replace(year=fecha_año)
        range1_max = datetime.strptime("31-Dec", "%d-%b").replace(year=fecha_año)
        range2_min = datetime.strptime("1-Jan", "%d-%b").replace(year=fecha_año)
        range2_max = datetime.strptime("3-Mar", "%d-%b").replace(year=fecha_año)
        range3_min = datetime.strptime("15-Jul", "%d-%b").replace(year=fecha_año)
        range3_max = datetime.strptime("31-Jul", "%d-%b").replace(year=fecha_año)
        range4_min = datetime.strptime("11-Sep", "%d-%b").replace(year=fecha_año)
        range4_max = datetime.strptime("30-Sep", "%d-%b").replace(year=fecha_año)

        if (
            (fecha >= range1_min and fecha <= range1_max)
            or (fecha >= range2_min and fecha <= range2_max)
            or (fecha >= range3_min and fecha <= range3_max)
            or (fecha >= range4_min and fecha <= range4_max)
        ):
            return 1
        else:
            return 0

    @staticmethod
    def _get_period_day(date):
        date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
        morning_min = datetime.strptime("05:00", "%H:%M").time()
        morning_max = datetime.strptime("11:59", "%H:%M").time()
        afternoon_min = datetime.strptime("12:00", "%H:%M").time()
        afternoon_max = datetime.strptime("18:59", "%H:%M").time()
        evening_min = datetime.strptime("19:00", "%H:%M").time()
        evening_max = datetime.strptime("23:59", "%H:%M").time()
        night_min = datetime.strptime("00:00", "%H:%M").time()
        night_max = datetime.strptime("4:59", "%H:%M").time()

        if date_time > morning_min and date_time < morning_max:
            return "mañana"
        elif date_time > afternoon_min and date_time < afternoon_max:
            return "tarde"
        elif (date_time > evening_min and date_time < evening_max) or (
            date_time > night_min and date_time < night_max
        ):
            return "noche"

    @staticmethod
    def _get_min_diff(data):
        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        return ((fecha_o - fecha_i).total_seconds()) / 60

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Check if necessary columns are present
        necessary_columns = ["OPERA", "TIPOVUELO", "MES"]
        for col in necessary_columns:
            if col not in data.columns:
                raise ValueError(f"Missing necessary column: {col}")
        # Feature engineering
        if target_column:
            data["period_day"] = data["Fecha-I"].apply(self._get_period_day)
            data["high_season"] = data["Fecha-I"].apply(self._is_high_season)
            data["min_diff"] = data.apply(self._get_min_diff, axis=1)

            data[target_column] = np.where(
                data["min_diff"] > self._threshold_in_minutes, 1, 0
            )
        # Convert categorical columns to dummy variables
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )
        # Ensure all top features columns are present in the features dataframe
        for col in self._top_features:
            if col not in features.columns:
                features[col] = 0

        features = features[self._top_features]

        # If target_column is provided, split data into features and target
        return (features, data[[target_column]]) if target_column else features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        logger.info(f"Class balance: {self._scale}")

        # Initialize and train the XGBoost model
        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=self._scale
        )
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if not self._model:
            raise ValueError("Model hasn't been trained yet!")
        return self._model.predict(features).tolist()

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if self._model:
            dump(self._model, filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            raise ValueError("No model has been trained yet. Cannot save.")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        self._model = load(filepath)
        logger.info(f"Model loaded from {filepath}")
