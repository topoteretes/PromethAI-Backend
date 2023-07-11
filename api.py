from llm_chains.chains import Agent
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import re
import json
import logging
import os
import uvicorn
from fastapi import Request
import yaml
from fastapi import HTTPException
CANNED_RESPONSES = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)

logger = logging.getLogger(__name__)
from dotenv import load_dotenv


load_dotenv()


app = FastAPI(debug=True)

from typing import Optional
# class Payload(BaseModel):
#     user_id: str
#     session_id: str
#     prompt_struct: Optional[dict] = None
#     model_speed: str
#     category: Optional[str] = None
#     from_: Optional[str] = None
#     to: Optional[str] = None
#     prompt: Optional[str] = None

class Payload(BaseModel):
    payload: Dict[str, Any]
class ImageResponse(BaseModel):
    success: bool
    message: str









# #
def splitter(t):
    lst = t.split("=")
    if len(lst) >= 2:
        key = lst[0].strip()
        value = lst[1].strip()

        # Separate key with camel case or underscore
        key = re.sub(r"(?<=[a-z])(?=[A-Z])|_", " ", key)
        key = key.lower()

        return {
            "category": key,
            "options": [{"category": value, "options": []}],
            "preference": [value],
        }
    else:
        return None
from auth.cognito.JWTBearer import JWTBearer
from auth.auth import jwks
auth = JWTBearer(jwks)

from fastapi import Depends


@app.get("/", )
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Hello, World, I am alive!"}

@app.get("/health",dependencies=[Depends(auth)])
def health_check():
    """
    Health check endpoint that returns the server status.
    """
    return {"status": "OK"}
@app.post("/test", response_model=dict)
async def test(request_data: Payload) -> dict:
    """
    Endpoint to clear the cache.

    Parameters:
    request_data (Payload): The request data containing the user and session IDs.

    Returns:
    dict: A dictionary with a message indicating the cache was cleared.
    """
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    try:
        agent._simple_test()
        return JSONResponse(content={"response": "Test"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/clear-cache", response_model=dict,dependencies=[Depends(auth)])
async def clear_cache(request_data: Payload) -> dict:
    """
    Endpoint to clear the cache.

    Parameters:
    request_data (Payload): The request data containing the user and session IDs.

    Returns:
    dict: A dictionary with a message indicating the cache was cleared.
    """
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    try:
        agent.clear_cache()
        return JSONResponse(content={"response": "Cache cleared"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/correct-prompt-grammar", response_model=dict,dependencies=[Depends(auth)])
async def prompt_to_correct_grammar(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    logging.info("Correcting grammar %s", json_payload["prompt_source"])

    output = agent.prompt_correction(json_payload["prompt_source"], model_speed= json_payload["model_speed"])
    return JSONResponse(content={"response": {"result": json.loads(output)}})


@app.post("/action-add-zapier-calendar-action", response_model=dict,dependencies=[Depends(auth)])
async def action_add_zapier_calendar_action(
    request: Request, request_data: Payload
) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    # Extract the bearer token from the header
    auth_header = request.headers.get("Authorization")
    if auth_header:
        bearer_token = auth_header.replace("Bearer ", "")
    else:
        bearer_token = None
    outcome = agent.add_zapier_calendar_action(
        prompt_base=json_payload["prompt_base"],
        token=bearer_token,
        model_speed=json_payload["model_speed"],
    )
    return JSONResponse(content={"response": outcome})


@app.post("/prompt-to-choose-meal-tree", response_model=dict,dependencies=[Depends(auth)])
async def prompt_to_choose_meal_tree(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.prompt_to_choose_tree(
        json_payload["prompt"],
        model_speed=json_payload["model_speed"],
        assistant_category="food",
    )
    logging.info("HERE IS THE CHAIN RESULT %s", output)
    result = json.dumps(
        {"results": list(map(splitter, output.replace('"', "").split(";")))}
    )
    return JSONResponse(content={"response": json.loads(result)})

def create_endpoint_with_resources(category: str, solution_type: str, prompt: str, json_example: str, *args, **kwargs):
    class Payload(BaseModel):
        payload: Dict[str, Any]

    @app.post(f"/chatbot/{category}", response_model=dict,dependencies=[Depends(auth)])
    async def prompt_to_choose_tree(request_data: Payload) -> dict:
        json_payload = request_data.payload
        from bots.bot_extension import AppAgent
        agent =  AppAgent()
        agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
        output = agent.query(
            json_payload["prompt"]
        )
        logging.info("HERE IS THE CHAIN RESULT %s", output)


        return JSONResponse(content={"response": output})




def create_endpoint(category: str, solution_type: str, prompt: str, json_example: str, *args, **kwargs):
    class Payload(BaseModel):
        payload: Dict[str, Any]

    @app.post(f"/{category}/prompt-to-choose-tree", response_model=dict,dependencies=[Depends(auth)])
    async def prompt_to_choose_tree(request_data: Payload) -> dict:
        json_payload = request_data.payload
        agent = Agent()
        agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
        output = agent.prompt_to_choose_tree(
            json_payload["prompt"],
            model_speed=json_payload["model_speed"],
            assistant_category=category,
        )
        logging.info("HERE IS THE CHAIN RESULT %s", output)
        result = json.dumps(
            {"results": list(map(splitter, output.replace('"', "").split(";")))}
        )

        return JSONResponse(content={"response": json.loads(result)})

    #this doesn't work
    @app.post(f"/{category}/prompt-to-decompose-categories", response_model=dict,dependencies=[Depends(auth)])
    async def prompt_to_decompose_categories(request_data: Payload) -> dict:
        json_payload = request_data.payload
        agent = Agent()
        agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
        output = await agent.prompt_decompose_to_tree_categories(
            json_payload["prompt_struct"],
            assistant_category=category,
            model_speed=json_payload["model_speed"],
        )
        return JSONResponse(content={"response": output})

    @app.post(f"/{category}/update-agent-summary/{solution_type}", response_model=dict,dependencies=[Depends(auth)])
    async def update_agent_summary(request_data: Payload) -> dict:
        json_payload = request_data.payload
        agent = Agent()
        agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
        output = await agent.update_agent_summary(
            model_speed=json_payload["model_speed"]
        )

        return {"response": output}

    @app.post(f"/{category}/prompt-to-update-tree", response_model=dict,dependencies=[Depends(auth)])
    async def prompt_to_update_tree(request_data: Payload) -> dict:
        json_payload = request_data.payload
        agent = Agent()
        agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
        output = agent.prompt_to_update_meal_tree(
            json_payload["category"],
            json_payload["from"],
            json_payload["to"],
            model_speed=json_payload["model_speed"],
        )

        print("HERE IS THE OUTPUT", output)
        return JSONResponse(content={"response": output})

    @app.post(f"/{category}/fetch-user-summary/{solution_type}", response_model=dict,dependencies=[Depends(auth)])
    async def fetch_user_summary(request_data: Payload) -> dict:
        json_payload = request_data.payload
        agent = Agent()
        agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
        output = agent.fetch_user_summary(model_speed=json_payload["model_speed"])

        return {"response": output}

    @app.post(f"/{category}/request/{solution_type}", response_model=dict,dependencies=[Depends(auth)])
    async def solution_request(request_data: Payload) -> dict:
        json_payload = request_data.payload
        agent = Agent()
        agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
        # method_to_call = getattr(agent, f"{solution_type}_generation")
        output = agent.solution_generation(json_payload["prompt"], prompt_template=prompt, json_example=json_example, model_speed="slow")
        output = output.replace("'", '"')
        return JSONResponse(content={"response": json.loads(output)})


# Load categories from a yaml file
with open('assistant_templates.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Create an endpoint for each category and solution type
for role in ['assistant', 'chatbot']:
    # If the role is 'assistant'
    if role == 'assistant':
        # Iterate through the categories and solution_types
        for category in data[role]['categories']:
            for solution_type in category['solution_types']:
                create_endpoint(category['name'], solution_type['name'], solution_type['prompt'], json.loads(solution_type['json_example']))
    # If the role is 'chatbot'
    elif role == 'chatbot':
        pass
        # Iterate through the categories and resources
        # for category in data[role]['categories']:
        #         create_endpoint_with_resources(category['name'])
@app.post("/prompt-to-decompose-meal-tree-categories", response_model=dict,dependencies=[Depends(auth)])
async def prompt_to_decompose_meal_tree_categories(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = await agent.prompt_decompose_to_tree_categories(
        json_payload["prompt_struct"],
        assistant_category="food",
        model_speed=json_payload["model_speed"],
    )

    return JSONResponse(content={"response": output})


@app.post("/correct-prompt-grammar", response_model=dict,dependencies=[Depends(auth)])
async def prompt_to_correct_grammar(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    logging.info("Correcting grammar %s", json_payload["prompt_source"])

    output = agent.prompt_correction(json_payload["prompt_source"], model_speed= json_payload["model_speed"])
    return JSONResponse(content={"response": {"result": json.loads(output)}})


@app.post("/prompt-to-update-meal-tree", response_model=dict,dependencies=[Depends(auth)])
async def prompt_to_update_meal_tree(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.prompt_to_update_meal_tree(
        json_payload["category"],
        json_payload["from"],
        json_payload["to"],
        model_speed=json_payload["model_speed"],
    )
    print("HERE IS THE OUTPUT", output)
    return JSONResponse(content={"response": output})


@app.post("/fetch-user-summary", response_model=dict,dependencies=[Depends(auth)])
async def fetch_user_summary(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.fetch_user_summary(model_speed=json_payload["model_speed"])
    print("HERE IS THE OUTPUT", output)
    return JSONResponse(content={"response": output})


@app.post("/recipe-request", response_model=dict,dependencies=[Depends(auth)])
async def recipe_request(request_data: Payload) -> dict:
    if CANNED_RESPONSES:
        with open("fixtures/recipe_response.json", "r") as f:
            json_data = json.load(f)
            stripped_string_dict = {"response": json_data}
            return JSONResponse(content=stripped_string_dict)

    json_payload = request_data.payload
    # factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])

    output = agent.solution_generation(json_payload["prompt"], model_speed="slow", prompt_template=None, json_example=None)
    output = str(output).replace("'", '"')
    return JSONResponse(content={"response": json.loads(output)})


# @app.post("/restaurant-request", response_model=dict)
# async def restaurant_request(request_data: Payload) -> dict:
#     json_payload = request_data.payload
#     agent = Agent()
#     agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
#     output = agent.restaurant_generation(json_payload["prompt"], model_speed="slow")
#     return JSONResponse(content={"response": {"restaurants": output}})


# @app.post("/delivery-request", response_model=dict)
# async def delivery_request(request_data: Payload) -> dict:
#     json_payload = request_data.payload
#     # factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
#     agent = Agent()
#     agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
#     output = await agent.delivery_generation( json_payload["prompt"], zipcode=json_payload["zipcode"], model_speed="slow")
#     print("HERE IS THE OUTPUT", output)
#     return JSONResponse(content={"response": {"url": output}})


@app.post("/voice-input", response_model=dict,dependencies=[Depends(auth)])
async def voice_input(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.voice_text_input(
        query=json_payload["query"], model_speed=json_payload["model_speed"]
    )
    return JSONResponse(content={"response": output})





def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the API server using uvicorn.

    Parameters:
    host (str): The host for the server.
    port (int): The port for the server.
    """
    try:
        logger.info(f"Starting server at {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.exception(f"Failed to start server: {e}")
        # Here you could add any cleanup code or error recovery code.


if __name__ == "__main__":
    start_api_server()
