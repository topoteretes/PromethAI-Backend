#Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client

import dlt
from langchain import PromptTemplate, LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import weaviate
import os
import json

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.retrievers import WeaviateHybridSearchRetriever
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain.vectorstores import Weaviate
import uuid
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

embeddings = OpenAIEmbeddings()

from deep_translator import (GoogleTranslator)

def _convert_pdf_to_document(path: str = None):

    """Convert a PDF document to a Document object"""
    if path is None:
        raise ValueError("A valid path to the document must be provided.")

    loader = PyPDFLoader(path)
    pages = loader.load_and_split()

    print("PAGES", pages[0])

    # Parse metadata from the folder path
    path_parts = Path(path).parts
    personal_receipts_index = path_parts.index("personal_receipts")
    metadata_parts = path_parts[personal_receipts_index+1:]

    documents = []
    for page in pages:
        translation = GoogleTranslator(source='auto', target='en').translate(text=page.page_content)
        documents.append(
            Document(
                metadata={
                    "title": "Ticket: Public Transport",
                    "country": metadata_parts[1],
                    "year": metadata_parts[0],
                    "author": str(uuid.uuid4()),
                    "source": "/".join(metadata_parts),
                },
                page_content=translation,
            )
        )
    print(documents)

    return documents



def _init_weaviate():
    """Initialize weaviate client and retriever"""
    auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get('WEAVIATE_API_KEY'))
    client = weaviate.Client(
        url='https://my-vev-index-o4qitptw.weaviate.network',
        auth_client_secret=auth_config,

        additional_headers={
            "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')
        }
    )
    retriever = WeaviateHybridSearchRetriever(
        client=client,
        index_name="PDFloader",
        text_key="text",
        attributes=[],
        embedding=embeddings,
        create_schema_if_missing=True,
    )

    return retriever
def load_to_weaviate(document_path=None):
    """Load documents to weaviate"""
    retriever =_init_weaviate()

    docs = _convert_pdf_to_document(document_path)

    return retriever.add_documents(docs)


def get_from_weaviate(query=None, filters=None):
    """Get documents from weaviate, pass dict as filters"""
    """  {
        'path': ['year'],
        'operator': 'Equal',
        'valueText': '2017*'     }"""
    retriever = _init_weaviate()

    output = retriever.get_relevant_documents(
        query,
        score=True,
    )
    if filters:
        output = retriever.get_relevant_documents(
            query,
            score=True,
            where_filter=filters
        )

    return output


def delete_from_weaviate(query=None, filters=None):
    """Delete documents from weaviate, pass dict as filters"""
    """  {
        'path': ['year'],
        'operator': 'Equal',
        'valueText': '2017*'     }"""
    auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get('WEAVIATE_API_KEY'))
    client = weaviate.Client(
        url='https://my-vev-index-o4qitptw.weaviate.network',
        auth_client_secret=auth_config,

        additional_headers={
            "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')
        }
    )
    client.batch.delete_objects(
        class_name='PDFloader',
        # Same `where` filter as in the GraphQL API
        where={
            'path': ['year'],
            'operator': 'Equal',
            'valueText': '2017*'
        },
    )

    return "Success"


llm = ChatOpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="gpt-4-0613",
        )



def infer_schema_from_text(text: str):
    """Infer schema from text"""

    prompt_ = """ You are a json schema master. Create a JSON schema based on the following data and don't write anything else: {prompt} """

    complete_query = PromptTemplate(
    input_variables=["prompt"],
    template=prompt_,
)

    chain = LLMChain(
        llm=llm, prompt=complete_query, verbose=True
    )
    chain_result = chain.run(prompt=text).strip()
    # json_data = json.dumps(chain_result)
    return chain_result
#
#
# # print(infer_schema_from_text(output[0].page_content))


def load_json_or_infer_schema(file_path):
    """Load JSON schema from file or infer schema from text"""
    try:
        # Attempt to load the JSON file
        with open(file_path, 'r') as file:
            json_schema = json.load(file)
        return json_schema
    except FileNotFoundError:
        # If the file doesn't exist, run the specified function
        output = _convert_pdf_to_document(path="../document_store/personal_receipts/2017/de/public transport/3ZCCCW.pdf")
        json_schema = infer_schema_from_text(output[0].page_content)
        return json_schema




def ai_function(prompt=None, json_schema=None):
    """AI function to convert unstructured data to structured data"""
    # Here we define the user prompt and the structure of the output we desire
    # prompt = output[0].page_content

    prompt_msgs = [
        SystemMessage(
            content="You are a world class algorithm converting unstructured data into structured data."
        ),
        HumanMessage(content="Convert unstructured data to structured data:"),
        HumanMessagePromptTemplate.from_template("{input}"),
        HumanMessage(content="Tips: Make sure to answer in the correct format"),
    ]
    prompt_ = ChatPromptTemplate(messages=prompt_msgs)
    chain = create_structured_output_chain(json_schema , prompt=prompt_, llm=llm, verbose=True)
    output = chain.run(input = prompt, llm=llm)
    yield output

file_path = 'ticket_schema.json'
json_schema = load_json_or_infer_schema(file_path)
# # Here we initialize DLT pipeline and export the data to duckdb
pipeline = dlt.pipeline(pipeline_name ="train_ticket", destination='duckdb',  dataset_name='train_ticket_data')
output = _convert_pdf_to_document(path="../document_store/personal_receipts/2017/de/public transport/3ZCCCW.pdf")
info = pipeline.run(data =ai_function(output[0].page_content, json_schema))
print(info)

