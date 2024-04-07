from src.services.rate_limiter import _rate_limit_exceeded_handler
import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
import json


@pytest.mark.asyncio
async def test_rate_limit_exceeded_handler(mocker):
    request = Request({"type": "http"})
    exc = mocker.MagicMock()
    response = await _rate_limit_exceeded_handler(request, exc)
    response_body = json.loads(response.body.decode())
    assert response.status_code == 429
    assert 'detail' in response_body
    assert response_body['detail'] == "Rate limit exceeded"
