import unittest

import numpy as np
from fastapi.testclient import TestClient
from challenge import app
from unittest.mock import patch, Mock


class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_should_get_predict(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "N",
                    "MES": 3
                }
            ]
        }
        with patch("xgboost.XGBClassifier.predict", return_value=np.array([0])):
            response = self.client.post("/predict", json=data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"predict": [0]})

    def test_should_failed_unkown_column_1(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "N",
                    "MES": 13
                }
            ]
        }
        with patch("xgboost.XGBClassifier.predict", return_value=np.array([0])):
            response = self.client.post("/predict", json=data)
            self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_2(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "O",
                    "MES": 13
                }
            ]
        }
        with patch("xgboost.XGBClassifier.predict", return_value=np.array([0])):
            response = self.client.post("/predict", json=data)
            self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_3(self):
        data = {
            "flights": [
                {
                    "OPERA": "Argentinas",
                    "TIPOVUELO": "O",
                    "MES": 13
                }
            ]
        }
        with patch("xgboost.XGBClassifier.predict", return_value=np.array([0])):
            response = self.client.post("/predict", json=data)
            self.assertEqual(response.status_code, 400)

    def test_api_missing_columns(self):
        data = {
            "flights": [
                {
                    "TIPOVUELO": "N",
                    "MES": 3
                }
            ]
        }
        with patch("xgboost.XGBClassifier.predict", return_value=np.array([0])):
            response = self.client.post("/predict", json=data)
            self.assertEqual(response.status_code, 422)
            self.assertIn(response.text, '{"detail":[{"type":"missing","loc":["body","flights",0,"OPERA"],"msg":"Field required","input":{"TIPOVUELO":"N","MES":3},"url":"https://errors.pydantic.dev/2.3/v/missing"}]}')

    def test_api_incorrect_data_types(self):
        data = {
            "flights": [
                {
                    "OPERA": 12345,  # Using an integer instead of a string
                    "TIPOVUELO": "N",
                    "MES": 3
                }
            ]
        }
        with patch("xgboost.XGBClassifier.predict", return_value=np.array([0])):
            response = self.client.post("/predict", json=data)
            self.assertEqual(response.status_code, 422)
            self.assertIn(response.text, '{"detail":[{"type":"string_type","loc":["body","flights",0,"OPERA"],"msg":"Input should be a valid string","input":12345,"url":"https://errors.pydantic.dev/2.3/v/string_type"}]}')

    def test_api_out_of_range_values(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "N",
                    "MES": 15  # Out of range value
                }
            ]
        }
        with patch("xgboost.XGBClassifier.predict", return_value=np.array([0])):
            response = self.client.post("/predict", json=data)
            self.assertEqual(response.status_code, 400)
            self.assertIn(response.text, '{"detail":"Invalid MES value."}')

    def test_api_unseen_categorical_values(self):
        data = {
            "flights": [
                {
                    "OPERA": "Unknown Airline",
                    "TIPOVUELO": "N",
                    "MES": 3
                }
            ]
        }
        with patch("xgboost.XGBClassifier.predict", return_value=np.array([0])):
            response = self.client.post("/predict", json=data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"predict": [0]})
