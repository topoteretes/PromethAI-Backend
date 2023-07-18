import os
from dotenv import load_dotenv
from api import start_api_server

# API_ENABLED = os.environ.get("API_ENABLED", "False").lower() == "true"
import boto3

environment = os.getenv("env", "dev")



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

    if os.path.exists(env_file_path):
        print(f"The .env file is located at: {os.path.abspath(env_file_path)}")
        load_dotenv()
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

        print("LEN OF PINECONE_API_KEY", len(PINECONE_API_KEY))
    else:
        print("The .env file was not found.")
    return "Success in loading env files"


env_file = ".env"
if os.path.exists(env_file):
    # Load default environment variables (.env)
    load_dotenv()
    print("Talk to the AI!")


else:
    secrets = fetch_secret(
        f"promethai-{environment}-backend-secretso-promethaijs-dotenv",
        "eu-west-1",
        ".env",
    )
    if secrets:
        print(secrets)
    load_dotenv()
import requests

# Fetch the task metadata endpoint from the ECS container environment variables
metadata_endpoint = requests.get('http://169.254.170.2/v2/metadata').json()['TaskARN']

# Extract the cluster and task ARN from the metadata endpoint
cluster_arn = metadata_endpoint.split('/')[1]
task_arn = metadata_endpoint.split('/')[3]

# Check if "dev" is present in the task ARN
if "dev" in task_arn:
    # Fetch the secret
    secrets = fetch_secret(
        f"promethai-dev-backend-secretso-promethaijs-dotenv",
        "eu-west-1",
        ".env",
    )
    load_dotenv()
elif "prd" in task_arn:
    # Fetch the secret
    secrets = fetch_secret(
        f"promethai-prd-backend-secretso-promethaijs-dotenv",
        "eu-west-1",
        ".env",
    )
    load_dotenv()
