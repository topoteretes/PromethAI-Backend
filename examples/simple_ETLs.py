#note, you need to install dlt, langchain, and duckdb
#pip install dlt
#pip install langchain
#pip install duckdb
#pip install python-dotenv
#pip install openai
#you also need a .env file with your openai api key

from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import os
import dlt
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
llm = ChatOpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4-0613",
        )
@dlt.resource(name='output', write_disposition='replace')
def ai_function():
    # Here we define the user prompt and the structure of the output we desire
    prompt = "I want to eat something very healthy and tasty."
    json_schema = {
        "title": "Recipe name",
        "description": "Recipe description",
        "type": "object",
        "properties": {
            "ingredients": {"title": "Ingredients", "description": "Detailed ingredients", "type": "string"},
            "steps": {"title": "Cooking steps", "description": "Detailed cooking steps", "type": "string"}
        },
        "required": ["ingredients", "steps"],
    }
    prompt_msgs = [
        SystemMessage(
            content="You are a world class algorithm for creating recipes"
        ),
        HumanMessage(content="Create a food recipe based on the following prompt:"),
        HumanMessagePromptTemplate.from_template("{input}"),
        HumanMessage(content="Tips: Make sure to answer in the correct format"),
    ]
    prompt_ = ChatPromptTemplate(messages=prompt_msgs)
    chain = create_structured_output_chain(json_schema, prompt=prompt_, llm=llm, verbose=True)
    output = chain.run(input = prompt, llm=llm)
    yield output

# Here we initialize DLT pipeline and export the data to duckdb
pipeline = dlt.pipeline(pipeline_name ="recipe", destination='duckdb',  dataset_name='recipe_data')
info = pipeline.run(data =ai_function())
print(info)
