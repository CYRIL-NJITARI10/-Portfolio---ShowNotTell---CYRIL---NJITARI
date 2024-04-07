import pytest
from fastapi.testclient import TestClient
from src.app import get_application

@pytest.fixture(scope="module")
def test_client():
    app = get_application()
    with TestClient(app) as test_client:
        yield test_client


