from tests.conftest import *
from pytest_mock import mocker
from unittest.mock import patch
from unittest.mock import MagicMock
import pandas as pd

def test_download_iris_dataset(test_client, mocker):
    mocker.patch("opendatasets.download", MagicMock())
    response = test_client.get("/download-iris")
    assert response.status_code == 200
    # Assurez-vous que la réponse contient un message de succès
    assert "message" in response.json()


def test_load_iris_dataset(test_client, mocker):
    mocker.patch("pandas.read_csv", MagicMock(return_value=pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})))
    response = test_client.get("/load-iris")
    assert response.status_code == 200
    assert len(response.json()) > 0


def test_split_iris_dataset(test_client):
    iris_data_list = [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "setosa"},
        {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2, "species": "setosa"},
    ]
    response = test_client.post("/split-iris", json=iris_data_list)
    assert response.status_code == 200
    assert "train" in response.json()
    assert "test" in response.json()
