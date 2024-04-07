import pytest
from unittest.mock import MagicMock, patch
from src.services.training import train_iris_model


def test_train_iris_model(mocker):
    X_train = [[0, 1], [2, 3]]
    y_train = [0, 1]

    mocker.patch('src.services.training.load_model_parameters', return_value={'n_estimators': 100, 'criterion': 'gini'})

    mocker.patch('src.services.training.os.path.isdir', return_value=False)
    mocker.patch('src.services.training.os.makedirs')

    mock_joblib_dump = mocker.patch('src.services.training.joblib.dump')

    model = train_iris_model(X_train, y_train)

    assert model is not None
    mock_joblib_dump.assert_called_once()

