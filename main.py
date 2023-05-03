
import os
from dotenv import load_dotenv
from api import start_api_server
API_ENABLED = os.environ.get("API_ENABLED", "False").lower() == "true"
import boto3


def fetch_secret(secret_name, region_name, env_file_path):
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        response = client.get_secret_value(SecretId=secret_name)
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        return None

    if "SecretString" in response:
        secret = response["SecretString"]
    else:
        secret = response["SecretBinary"]
    with open(env_file_path, "w") as env_file:
        env_file.write(secret)
    print("HERE IS THE SECRET", secret)
    return "Success"


env_file = ".env"
if os.path.exists(env_file):
    # Load default environment variables (.env)
    load_dotenv()
    print("Talk to the AI!")
    # if API_ENABLED:
    # Run FastAPI application
    start_api_server()

else:
    secrets = fetch_secret("promethai-dev-backend-secretso-promethaijs-dotenv", "eu-west-1", ".env")
    if secrets:
        print(secrets)
    load_dotenv()
    start_api_server()
