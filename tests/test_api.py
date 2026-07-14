from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_predict_endpoint_returns_prediction():
    response = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_charge": 79.99,
            "support_tickets": 3,
            "contract_type": "month-to-month",
            "age": 34,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert "churn_probability" in data
    assert "predicted_label" in data
    assert 0 <= data["churn_probability"] <= 1
    assert data["predicted_label"] in [0, 1]


def test_predict_endpoint_rejects_invalid_contract_type():
    response = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_charge": 79.99,
            "support_tickets": 3,
            "contract_type": "invalid-contract",
            "age": 34,
        },
    )

    assert response.status_code == 422