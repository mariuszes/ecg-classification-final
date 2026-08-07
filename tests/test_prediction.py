import os

import httpx


BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")


def test_predict_row():
    response = httpx.post(
        f"{BASE_URL}/predict-row",
        json={
            "row_index": 0
        },
        timeout=30,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["row_index"] == 0
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["normal", "abnormal"]
    assert data["true_target"] in [0, 1]
    assert data["true_label"] in ["normal", "abnormal"]
    assert 0 <= data["probability_abnormal"] <= 1