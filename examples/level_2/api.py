from level_2_pdf_vectorstore__dlt_contracts import ShortTermMemory
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



# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)

logger = logging.getLogger(__name__)
from dotenv import load_dotenv


load_dotenv()


app = FastAPI(debug=True)


from fastapi import Depends


class Payload(BaseModel):
    payload: Dict[str, Any]
class ImageResponse(BaseModel):
    success: bool
    message: str




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



# @app.post("/testbot", response_model=Dict[str, Any])
# async def test(request_data: Payload) -> Dict[str, Any]:
#     """
#     Endpoint to clear the cache.
#
#     Parameters:
#     request_data (Payload): The request data containing the user and session IDs.
#
#     Returns:
#     dict: A dictionary with a message indicating the cache was cleared.
#     """
#     json_payload = request_data.payload
#
#     try:
#         # Instantiate AppAgent and call manage_resources
#         app_agent = AppAgent(user_id=json_payload["user_id"])
#         app_agent.manage_resources("add", "web_page", "https://nav.al/agi")
#         return JSONResponse(content={"response": "Test"}, status_code=200)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/clear-cache", response_model=dict)
# async def clear_cache(request_data: Payload) -> dict:
#     """
#     Endpoint to clear the cache.
#
#     Parameters:
#     request_data (Payload): The request data containing the user and session IDs.
#
#     Returns:
#     dict: A dictionary with a message indicating the cache was cleared.
#     """
#     json_payload = request_data.payload
#     agent = Agent()
#     agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
#     try:
#         agent.clear_cache()
#         return JSONResponse(content={"response": "Cache cleared"}, status_code=200)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
# @app.post("/correct-prompt-grammar", response_model=dict)
# async def prompt_to_correct_grammar(request_data: Payload) -> dict:
#     json_payload = request_data.payload
#     agent = Agent()
#     agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
#     logging.info("Correcting grammar %s", json_payload["prompt_source"])
#
#     output = agent.prompt_correction(json_payload["prompt_source"], model_speed= json_payload["model_speed"])
#     return JSONResponse(content={"response": {"result": json.loads(output)}})


# @app.post("/action-add-zapier-calendar-action", response_model=dict,dependencies=[Depends(auth)])
# async def action_add_zapier_calendar_action(
#     request: Request, request_data: Payload
# ) -> dict:
#     json_payload = request_data.payload
#     agent = Agent()
#     agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
#     # Extract the bearer token from the header
#     auth_header = request.headers.get("Authorization")
#     if auth_header:
#         bearer_token = auth_header.replace("Bearer ", "")
#     else:
#         bearer_token = None
#     outcome = agent.add_zapier_calendar_action(
#         prompt_base=json_payload["prompt_base"],
#         token=bearer_token,
#         model_speed=json_payload["model_speed"],
#     )
#     return JSONResponse(content={"response": outcome})



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
