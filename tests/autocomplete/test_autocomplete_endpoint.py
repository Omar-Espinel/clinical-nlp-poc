import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

import api as api_module
from api import app

API_KEY = "test-key-123"


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", API_KEY)


@pytest.fixture
def mock_autocomplete():
    mock = MagicMock()
    mock.run = AsyncMock(return_value={
        "suggestions": [
            {
                "display": "Diabetes Mellitus",
                "completion": "Diabetes Mellitus",
                "snomed_code": "73211009",
                "category": "snomed",
                "confidence": 0.99,
                "match_type": "prefix",
                "deprecated": False,
            }
        ],
        "processing_time_ms": 45,
        "tier_used": "tier1",
        "fallback_active": False,
    })
    return mock


@pytest.fixture
def client(mock_autocomplete):
    original = api_module.autocomplete
    api_module.autocomplete = mock_autocomplete
    with TestClient(app) as c:
        yield c
    api_module.autocomplete = original


def test_autocomplete_requires_api_key(client):
    response = client.get("/v1/autocomplete", params={"q": "diab"})
    assert response.status_code == 401


def test_autocomplete_returns_200(client):
    response = client.get("/v1/autocomplete", params={"q": "diab"}, headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    data = response.json()
    assert "suggestions" in data
    assert "processing_time_ms" in data
    assert "tier_used" in data


def test_autocomplete_short_prefix_rejected_by_fastapi(client):
    response = client.get("/v1/autocomplete", params={"q": "di"}, headers={"X-API-Key": API_KEY})
    assert response.status_code == 422


def test_autocomplete_limit_too_high_rejected(client):
    response = client.get("/v1/autocomplete", params={"q": "diab", "limit": 99}, headers={"X-API-Key": API_KEY})
    assert response.status_code == 422


def test_autocomplete_empty_on_no_results(client):
    api_module.autocomplete.run = AsyncMock(return_value={
        "suggestions": [],
        "processing_time_ms": 10,
        "tier_used": "none",
        "fallback_active": False,
    })
    response = client.get("/v1/autocomplete", params={"q": "zzz"}, headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert response.json()["suggestions"] == []


def test_autocomplete_503_when_not_initialized(monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", API_KEY)
    # lifespan startup re-initializes `autocomplete`, so force it to None
    # *after* the app has started to exercise the not-ready 503 path.
    with TestClient(app) as c:
        monkeypatch.setattr(api_module, "autocomplete", None)
        response = c.get("/v1/autocomplete", params={"q": "diab"}, headers={"X-API-Key": API_KEY})
        assert response.status_code == 503


def test_autocomplete_rate_limit(monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", API_KEY)
    mock = MagicMock()
    mock.run = AsyncMock(return_value={
        "suggestions": [],
        "processing_time_ms": 10,
        "tier_used": "none",
        "fallback_active": False,
    })
    monkeypatch.setattr(api_module, "autocomplete", mock)
    api_module._autocomplete_rate_store.clear()
    with TestClient(app) as c:
        responses = [
            c.get("/v1/autocomplete", params={"q": "diab"}, headers={"X-API-Key": API_KEY}).status_code
            for _ in range(31)
        ]
        assert 429 in responses
