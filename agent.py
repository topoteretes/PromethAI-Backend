import openai
import os
import pinecone

from dotenv import load_dotenv
import nltk
from langchain.text_splitter import NLTKTextSplitter
from typing import Optional

# Download NLTK for Reading
nltk.download("punkt")
import subprocess
import datetime

# Initialize Text Splitter
text_splitter = NLTKTextSplitter(chunk_size=2500)
# Load default environment variables (.env)
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"

OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")


# Top matches length
k_n = 3

# initialize pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# initialize openAI
openai.api_key = OPENAI_API_KEY  # you can just copy and paste your key here if you want


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]
