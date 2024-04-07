from tests.conftest import *

def test_generate_token(test_client):
    response = test_client.post("/token", data={"username": "cyril", "password": "cyril"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_authentication_error_user_not_found(test_client):
    response = test_client.post("/token", data={"username": "invalid_user", "password": "cyril"})
    assert response.status_code == 401
    # Vérifiez le message d'erreur spécifique si applicable

def test_authentication_error_wrong_password(test_client):
    response = test_client.post("/token", data={"username": "cyril", "password": "wrong_password"})
    assert response.status_code == 401
    # Vérifiez le message d'erreur spécifique si applicable
