import requests
import json
from itertools import combinations

# Define the endpoint URL

endpoint_url = "http://promethai-dev-backend-alb-2012524587.eu-west-1.elb.amazonaws.com:8000/prompt-to-decompose-meal-tree-categories"

# Define the meal choice factors
meal_choice_factors = ["taste", "health", "cost", "cuisine", "hunger", "availability", "diet", "allergies", "time", "mood","calories"]

meal_choice_factors.sort()

print("Factors used for sorting: ", meal_choice_factors)


# Define the payload template
payload_template = {
    "payload": {
        "user_id": "123",
        "session_id": "471",
        "model_speed": "slow",
        "prompt_struct": ""
    }
}


headers = {
  'Content-Type': 'application/json'
}



# Generate combinations of three factors
factor_combinations = list(combinations(meal_choice_factors, 3))
# Get the total number of combinations
total_combinations = len(factor_combinations)

# Iterate through the combinations
for i, factors in enumerate(factor_combinations, 1):
    # Combine three factors with the prompt structure
    prompt_struct = ";".join([f"{factor}=Helsinki" for factor in factors])
    payload_template["payload"]["prompt_struct"] = prompt_struct

    # Convert payload template to JSON
    payload_json = json.dumps(payload_template)

    # Send the request to the endpoint
    response = requests.request("POST", endpoint_url, headers=headers, data=payload_json)

    # Print the response
    print(response.text)

    # Print the progress and remaining requests
    print(f"Progress: {i}/{total_combinations}")
    print(f"Requests remaining: {total_combinations - i}")
