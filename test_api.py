import pytest
from fastapi.testclient import TestClient
from main_chains import Agent
from .api import app

client = TestClient(app)

@pytest.fixture(scope="session")
def agent():
    yield Agent()

class TestRoutes:

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello, World, I am alive!"}

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}

    def test_generate_diet_sub_goal(self):
        payload = {
            "payload": {
                "user_id": "657",
                "session_id": "456",
                "model_speed": "slow",
                "factors": [
                    {"name": "Portion Control"},
                    {"name": "Cuisine"},
                    {"name": "Macronutrients"}
                ]
            }
        }
        response = client.post("/generate-diet-sub-goal", json=payload)
        assert response.status_code == 200
        response_body = response.json()

        # Check that the response structure is correct
        assert 'response' in response_body
        assert 'sub_goals' in response_body['response']

        # Check that all expected goals are present
        for goal in payload['payload']['factors']:
            found_goal = any(sub_goal['goal_name'] == goal['name'] for sub_goal in response_body['response']['sub_goals'])
            assert found_goal, f"Goal '{goal['name']}' not found in response"

    def test_generate_diet_goal(self):
        payload = {
            "payload": {
                "model_speed": "slow",
                "user_id": "659",
                "session_id": "458"
            }
        }
        response = client.post("/generate-diet-goal", json=payload)
        assert response.status_code == 200
        response_body = response.json()

        # Check that the response structure is correct
        assert 'response' in response_body
        assert 'goals' in response_body['response']

        # Example goals to check for in response
        example_goals = ["Cuisine", "Time to make", "Complexity", "Macros"]
        # Check that all example goals are present
        for goal in example_goals:
            assert goal in response_body['response']['goals'], f"Goal '{goal}' not found in response"

if __name__ == "__main__":
    pytest.main()
