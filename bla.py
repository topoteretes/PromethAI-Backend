import openai
import json

OPENAI_API_KEY = "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"


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_decision_nodes(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


# Step 1, send model the user query and what functions it has access to
def run_conversation():
    response = openai.ChatCompletion.create(
        api_key="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",
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        functions=[
            {
                "name": "get_decision_nodes",
                "description": "Get decision nodes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
        function_call="auto",
    )

    message = response["choices"][0]["message"]

    # Step 2, check if the model wants to call a function
    if message.get("function_call"):
        function_name = message["function_call"]["name"]

        # Step 3, call the function
        # Note: the JSON response from the model may not be valid JSON
        function_response = get_decision_nodes(
            location=message.get("location"),
            unit=message.get("unit"),
        )

        # Step 4, send model the info on the function call and function response
        second_response = openai.ChatCompletion.create(
            api_key="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",
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": "What is the weather like in boston?"},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        return second_response


print(run_conversation())
