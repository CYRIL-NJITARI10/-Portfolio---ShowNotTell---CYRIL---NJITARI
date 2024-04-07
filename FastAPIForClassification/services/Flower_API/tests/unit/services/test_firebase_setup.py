import pytest
from src.services.firebase_setup import initialize_firestore
from unittest.mock import MagicMock
from src.services.firebase_setup import create_firestore_collection


def test_initialize_firestore(mocker):
    mocker.patch('src.services.firebase_setup.credentials.Certificate', return_value=MagicMock())
    mocker.patch('src.services.firebase_setup.firebase_admin.initialize_app')
    mocker.patch('src.services.firebase_setup.firestore.client', return_value=MagicMock())

    db = initialize_firestore()

    assert db is not None


def test_create_firestore_collection(mocker):
    mock_db = MagicMock()
    create_firestore_collection(mock_db)

    mock_db.collection.assert_called_once_with('parameters')
    mock_db.collection().document.assert_called_once_with('model_params')
    mock_db.collection().document().set.assert_called_once_with({
        'n_estimators': 100,
        'criterion': 'gini'
    })
