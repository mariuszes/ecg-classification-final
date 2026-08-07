import os

import httpx


BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")


def test_model_info():
    response = httpx.get(
        f"{BASE_URL}/model-info",
        timeout=30,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model_uri"] == "models:/ecg_classifier@champion"
    assert data["model_type"] == "RandomForestClassifier"
    assert data["feature_count"] > 0