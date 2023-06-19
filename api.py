from chains import Agent
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import json
import logging
import os
import uvicorn

CANNED_RESPONSES=False

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)

logger = logging.getLogger(__name__)
from dotenv import load_dotenv

import re
load_dotenv()


app = FastAPI(debug=True)

class Payload(BaseModel):
    payload: Dict[str, Any]

class ImageResponse(BaseModel):
    success: bool
    message: str

@app.get("/")
async def root():
    return {"message": "Hello, World, I am alive!"}

def splitter(t):
    lst = t.split("=")
    if len(lst) >= 2:
        key = lst[0].strip()
        value = lst[1].strip()

        # Separate key with camel case or underscore
        key = re.sub(r'(?<=[a-z])(?=[A-Z])|_', ' ', key)
        key = key.lower()

        return {
            "category": key,
            "options": [
                {
                    "category": value,
                    "options": []
                }
            ],
            "preference": [value]
        }
    else:
        return None


@app.post("/clear-cache", response_model=dict)
async def clear_cache(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    agent.clear_cache()
    return JSONResponse(content={"response":"Cache cleared"})

@app.post("/prompt-to-choose-meal-tree", response_model=dict)
async def prompt_to_choose_meal_tree(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.prompt_to_choose_meal_tree(json_payload["prompt"], model_speed= json_payload["model_speed"],assistant_category="food")
    logging.info("HERE IS THE CHAIN RESULT %s", output)
    result = json.dumps({"results": list(map(splitter, output.replace('"', '').split(";")))})

    return JSONResponse(content={"response":json.loads(result)})


@app.post("/prompt-to-decompose-meal-tree-categories", response_model=dict)
async def prompt_to_decompose_meal_tree_categories(request_data: Payload)-> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = await agent.prompt_decompose_to_meal_tree_categories(json_payload["prompt_struct"], model_speed= json_payload["model_speed"])

    return JSONResponse(content={"response":output})


@app.post("/correct-prompt-grammar", response_model=dict)
async def prompt_to_correct_grammar(request_data: Payload)-> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    logging.info("Correcting grammar %s", json_payload["prompt_source"])
    output = agent.prompt_correction(json_payload["prompt_source"], model_speed= json_payload["model_speed"])
    return JSONResponse(content={"response": {"result": json.loads(output)}})

@app.post("/prompt-to-update-meal-tree", response_model=dict)
async def prompt_to_update_meal_tree(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.prompt_to_update_meal_tree(json_payload["category"], json_payload["from"], json_payload["to"], model_speed= json_payload["model_speed"])
    print("HERE IS THE OUTPUT", output)
    return JSONResponse(content={"response":output})

@app.post("/fetch-user-summary", response_model=dict)
async def fetch_user_summary(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.fetch_user_summary( model_speed= json_payload["model_speed"])
    print("HERE IS THE OUTPUT", output)
    return JSONResponse(content={"response":output})


@app.post("/recipe-request", response_model=dict)
async def recipe_request(request_data: Payload) -> dict:
    if CANNED_RESPONSES:
        with open('fixtures/recipe_response.json', 'r') as f:
            json_data = json.load(f)
            stripped_string_dict = {"response": json_data}
            return JSONResponse(content=stripped_string_dict)

    json_payload = request_data.payload
    # factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])

    output = agent.recipe_generation(json_payload["prompt"], model_speed="slow")
    return JSONResponse(content={"response":json.loads(output)});


@app.post("/restaurant-request", response_model=dict)
async def restaurant_request(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.restaurant_generation(json_payload["prompt"], model_speed="slow")
    return JSONResponse(content={"response":{"restaurants": output}});

# @app.post("/delivery-request", response_model=dict)
# async def delivery_request(request_data: Payload) -> dict:
#     json_payload = request_data.payload
#     # factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
#     agent = Agent()
#     agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
#     output = await agent.delivery_generation( json_payload["prompt"], zipcode=json_payload["zipcode"], model_speed="slow")
#     print("HERE IS THE OUTPUT", output)
#     return JSONResponse(content={"response": {"url": output}})

@app.post("/voice-input", response_model=dict)
async def voice_input(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.voice_text_input(query=json_payload["query"], model_speed= json_payload["model_speed"])
    return JSONResponse(content={"response":output})

@app.get("/health")
def health_check():
    return {"status": "OK"}

def start_api_server():
    # agent = establish_connection()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_api_server()
