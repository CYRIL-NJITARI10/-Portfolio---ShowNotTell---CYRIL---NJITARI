from unittest.mock import patch
from fastapi import status

def test_train_iris_model_authenticated(test_client):
    fake_token = 'fake_access_token'
    fake_user_data = {"username": "testuser"}

    # Patch the dependencies of the get_current_user function
    with patch('src.api.routes.authentication.oauth2_scheme', return_value=fake_token):
        with patch('src.api.routes.authentication.verify_token', return_value=fake_user_data):

            iris_data_list = [{
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
                "species": "setosa"
            }]

            headers = {
                "Authorization": f"Bearer {fake_token}"
            }

            response = test_client.post("/train-model", json=iris_data_list, headers=headers)

            assert response.status_code == status.HTTP_200_OK
            assert "message" in response.json()


def test_make_prediction(test_client):
    iris_data_list = [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]
    response = test_client.post("/predict", json=iris_data_list)
    assert response.status_code == 200
    assert "predictions" in response.json()
