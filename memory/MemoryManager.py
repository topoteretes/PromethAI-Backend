from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from Memory import Memory
import json
import os
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
llm = ChatOpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4-0613",
        )










class CognitiveTransformationProcess():

    def __init__(self, llm):
        self.llm = llm


    def episodic_manager(self, prompt):
        """The AI would need to comprehend the context of the stored information. For example, it might need to understand the user's past interactions or preferences to provide relevant information or actions. This process is similar to transforming raw data by applying business rules or logic to it."""


        #retrieve from STM here


        json_schema = {
            "title": "Episodic Memory",
            "description": "Detail of a specific episodic memory",
            "type": "object",
            "properties": {
                "event": {"title": "Event", "description": "Specific incident or event that occurred",
                          "type": "string"},
                "context": {"title": "Context", "description": "Situational detail around the event", "type": "string"},
                "emotions": {"title": "Emotions", "description": "Emotions associated with the event",
                             "type": "string"},
                "trackingCapture": {"title": "Tracking Capture", "description": "Tracking telemetry of the event",
                                    "type": "string"},
                "time": {"title": "Time", "description": "The time when the event occurred", "type": "string"}
            },
            "required": ["event", "context", "emotions", "sensePerception", "time"],
        }

        prompt_msgs = [
            SystemMessage(
                content="You are an advanced AI model capable of storing and recalling episodic memories"
            ),
            HumanMessage(content="Store an episodic memory based on the following details:"),
            HumanMessagePromptTemplate.from_template("{input}"),
            HumanMessage(content="Tips: Make sure to fill all the necessary details of the memory"),
        ]
        prompt_ = ChatPromptTemplate(messages=prompt_msgs)
        chain = create_structured_output_chain(json_schema, prompt=prompt_, llm=llm, verbose=True)
        output = chain.run(input=prompt, llm=llm)

        return output


    def semantic_content_manager(self, document_path:str="../document_store/nutrition/Human_Nutrition.pdf", knowledge_type:str=None):

        if knowledge_type == "book":

            loader = PyPDFLoader(document_path)
            pages = loader.load_and_split()
            memory=Memory()

            print("PAGES", pages[0])

            for page in pages:
                memory._update_semantic_memory(semantic_memory=page.page_content, knowledge_source="book")

        elif knowledge_type=="movie":
            pass


    def semantic_index_manager(self, prompt:str, knowledge_type:str):

        if knowledge_type =='book':
            folder_path = "index_templates"
            json_file_name = "book.json"
            # Join the folder path and file name to get the complete file path
            json_file_path = os.path.join(folder_path, json_file_name)

            # Load the JSON schema from the file
            with open(json_file_path, 'r') as file:
                json_schema = json.load(file)

        elif knowledge_type =='movie':
            folder_path = "index_templates"
            json_file_name = "movie.json"
            # Join the folder path and file name to get the complete file path
            json_file_path = os.path.join(folder_path, json_file_name)

            # Load the JSON schema from the file
            with open(json_file_path, 'r') as file:
                json_schema = json.load(file)

        prompt_msgs = [
            SystemMessage(
                content="You are an advanced AI model capable of storing and recalling semantic memories"
            ),
            HumanMessage(content="Store an semantic memory based on the following details:"),
            HumanMessagePromptTemplate.from_template("{input}"),
            HumanMessage(content="Tips: Make sure to fill all the necessary details of the memory"),
        ]
        prompt_ = ChatPromptTemplate(messages=prompt_msgs)
        chain = create_structured_output_chain(json_schema, prompt=prompt_, llm=llm, verbose=True)
        output = chain.run(input=prompt, llm=llm)
        memory = Memory()
        memory._update_semantic_memory(semantic_memory=output, knowledge_source="book")


    def buffer_manager(self, target:str, input:str):





        def retrieve_from_buffer():
            pass





    def event_manager(self):
        def insert_into_events(input: str):
            from connectors.duckdb_loader import DuckDB

            duckdb = DuckDB()

            data_load_metadata = duckdb.get_dlt_loads()

            memory = Memory()
            memory._update_event_memory(semantic_memory=data_load_metadata)
            pass



    def action_manager(self):
        pass










    def ltm_manager(self):
        pass





