import pytest
from fastapi.testclient import TestClient
from chains import Agent
from api import app

client = TestClient(app)


@pytest.fixture(scope="session")
def agent():
    yield Agent()


class TestRoutes:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello, World, I am alive!"}

    class TestRoutes:
        def test_root(self):
            response = client.get("/")
            assert response.status_code == 200
            assert response.json() == {"message": "Hello, World, I am alive!"}

        def test_health_check(self):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "OK"}

        def test_prompt_to_choose_meal_tree(self):
            payload = {
                "payload": {
                    "user_id": "657",
                    "session_id": "456",
                    "model_speed": "slow",
                    "prompt": "I want to eat healthy",
                }
            }
            response = client.post("/prompt-to-choose-meal-tree", json=payload)
            assert response.status_code == 200
            response_body = response.json()

            # Check that the response structure is correct
            assert "response" in response_body
            assert "results" in response_body["response"]

    def test_prompt_to_decompose_meal_tree_categories(self):
        payload = {
            "payload": {
                "user_id": "659",
                "session_id": "458",
                "model_speed": "slow",
                "prompt_struct": "taste=Helsinki;health=Helsinki;cost=Helsinki",
            }
        }
        response = client.post(
            "/prompt-to-decompose-meal-tree-categories", json=payload
        )
        assert response.status_code == 200
        response_body = response.json()

        # Check that the response structure is correct
        assert "response" in response_body
        assert "category" in response_body["response"]
        assert "options" in response_body["response"]

        # Check that the main category is 'location'
        assert response_body["response"]["category"] == "location"

        # Check that the options are correct
        options = response_body["response"]["options"]
        assert len(options) == 3  # There should be 3 options

        # Check that each option has a 'category' and 'options'
        for option in options:
            assert "category" in option
            assert "options" in option

            # Check that each sub-option has a 'category'
            for sub_option in option["options"]:
                assert "category" in sub_option


if __name__ == "__main__":
    pytest.main()
