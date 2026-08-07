import os

import httpx


BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")


def test_health_endpoint():
    response = httpx.get(f"{BASE_URL}/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"