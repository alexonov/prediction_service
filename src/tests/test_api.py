from falcon import testing, HTTP_OK, HTTP_BAD_REQUEST
import pytest
import json
from unittest.mock import MagicMock

from app import create_api
from tests.helpers import get_random_sample_dict
from controllers.model_controller import ModelContainer


EXPECTED_RESULT = {'prediction': [1, 2, 3, 4, 5, 6]}


@pytest.fixture(scope='module')
def client():
    model_container = MagicMock(spec=ModelContainer)
    model_container.predict.return_value = EXPECTED_RESULT['prediction']
    api = create_api(model_container)
    return testing.TestClient(api)


def test_api(client):
    result = client.simulate_get('/')
    assert result.status == HTTP_OK


def test_predict_valid_input_post(client):
    headers = {"Content-Type": "application/json"}
    body = json.dumps(get_random_sample_dict())
    result = client.simulate_post("/predict", headers=headers, body=body)

    assert result.status == HTTP_OK
    assert result.json == EXPECTED_RESULT


def test_predict_invalid_input_post(client):
    headers = {"Content-Type": "application/json"}
    body = json.dumps({})
    result = client.simulate_post("/predict", headers=headers, body=body)

    assert result.status == HTTP_BAD_REQUEST

