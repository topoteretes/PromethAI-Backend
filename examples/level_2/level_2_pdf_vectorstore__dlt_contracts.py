#Make sure to install the following packages: dlt, langchain, duckdb, python-dotenv, openai, weaviate-client

import dlt
from langchain import PromptTemplate, LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import weaviate
import os
import json
from marvin import ai_classifier
from enum import Enum
import marvin
import asyncio
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.retrievers import WeaviateHybridSearchRetriever
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain.tools import tool
from langchain.vectorstores import Weaviate
import uuid
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from langchain import OpenAI, LLMMathChain

import os

from datetime import datetime
import os
from datetime import datetime
from jinja2 import Template
from langchain import PromptTemplate, LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain.vectorstores import Weaviate
import weaviate
import uuid
load_dotenv()

class VectorDB:
    def __init__(self, user_id: str, index_name: str, memory_id:str, ltm_memory_id:str='00000', st_memory_id:str='0000', buffer_id:str='0000', db_type: str = "pinecone",  namespace:str = None):
        self.user_id = user_id
        self.index_name = index_name
        self.db_type = db_type
        self.namespace=namespace
        self.memory_id = memory_id
        self.ltm_memory_id = ltm_memory_id
        self.st_memory_id = st_memory_id
        self.buffer_id = buffer_id
        # if self.db_type == "pinecone":
        #     self.vectorstore = self.init_pinecone(self.index_name)
        if self.db_type == "weaviate":
            self.init_weaviate(namespace=self.namespace)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def init_pinecone(self, index_name):
        load_dotenv()
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
        PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "")
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
        pinecone.Index(index_name)
        vectorstore: Pinecone = Pinecone.from_existing_index(

            index_name=self.index_name,
            embedding=OpenAIEmbeddings(),
            namespace='RESULT'
        )
        return vectorstore

    def init_weaviate(self, namespace:str):
        embeddings = OpenAIEmbeddings()
        auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get('WEAVIATE_API_KEY'))
        client = weaviate.Client(
            url=os.environ.get('WEAVIATE_URL'),
            auth_client_secret=auth_config,

            additional_headers={
                "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')
            }
        )
        retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=namespace,
            text_key="text",
            attributes=[],
            embedding=embeddings,
            create_schema_if_missing=True,
        )
        return retriever

    def add_memories(self, observation: str,  page: str = "", source: str = ""):
        if self.db_type == "pinecone":
            # Update Pinecone memories here
            vectorstore: Pinecone = Pinecone.from_existing_index(
                index_name=self.index_name, embedding=OpenAIEmbeddings(), namespace=self.namespace
            )


            retriever = vectorstore.as_retriever()
            retriever.add_documents(
                [
                    Document(
                        page_content=observation,
                        metadata={
                            "inserted_at": datetime.now(),
                            "text": observation,
                            "user_id": self.user_id,
                            "page": page,
                            "source": source
                        },
                        namespace=self.namespace,
                    )
                ]
            )
        elif self.db_type == "weaviate":
            # Update Weaviate memories here
            retriever = self.init_weaviate( self.namespace)


            return retriever.add_documents([
                Document(
                    metadata={
                        "inserted_at": str(datetime.now()),
                        "text": observation,
                        "user_id": str(self.user_id),
                        "memory_id": str(self.memory_id),
                        "ltm_memory_id": str(self.ltm_memory_id),
                        "st_memory_id": str(self.st_memory_id),
                        "buffer_id": str(self.buffer_id),

                        # **source_metadata,
                    },
                    page_content=observation,
                )]
            )
    # def get_pinecone_vectorstore(self, namespace: str) -> pinecone.VectorStore:
    #     return Pinecone.from_existing_index(
    #         index_name=self.index, embedding=OpenAIEmbeddings(), namespace=namespace
    #     )

    def fetch_memories(self, observation: str, params = None):
        if self.db_type == "pinecone":
            # Fetch Pinecone memories here
            pass
        elif self.db_type == "weaviate":
            # Fetch Weaviate memories here
            """
            Get documents from weaviate.

            Args a json containing:
                query (str): The query string.
                path (list): The path for filtering, e.g., ['year'].
                operator (str): The operator for filtering, e.g., 'Equal'.
                valueText (str): The value for filtering, e.g., '2017*'.

            Example:
                get_from_weaviate(query="some query", path=['year'], operator='Equal', valueText='2017*')
            """
            retriever = self.init_weaviate(self.namespace)

            print(self.namespace)
            print(str(datetime.now()))

            # Retrieve documents with filters applied
            output = retriever.get_relevant_documents(
                observation,
                score=True,
                where_filter=params
            )

            return output

    def delete_memories(self, params: None):
        auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get('WEAVIATE_API_KEY'))
        client = weaviate.Client(
            url=os.environ.get('WEAVIATE_API_KEY'),
            auth_client_secret=auth_config,

            additional_headers={
                "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')
            }
        )
        client.batch.delete_objects(
            class_name=self.namespace,
            # Same `where` filter as in the GraphQL API
            where=params,
        )

    def update_memories(self):
        pass



class SemanticMemory:
    def __init__(self, user_id: str, memory_id:str, ltm_memory_id:str, index_name: str, db_type:str="weaviate", namespace:str="SEMANTICMEMORY"):
        # Add any semantic memory-related attributes or setup here
        self.user_id=user_id
        self.index_name = index_name
        self.namespace = namespace
        self.semantic_memory_id = str(uuid.uuid4())
        self.memory_id = memory_id
        self.ltm_memory_id = ltm_memory_id
        self.vector_db = VectorDB(user_id=user_id, memory_id= self.memory_id, ltm_memory_id = self.ltm_memory_id, index_name=index_name, db_type=db_type, namespace=self.namespace)
        self.db_type = db_type



    def _update_memories(self ,memory_id:str="None", semantic_memory: str="None") -> None:
        """Update semantic memory for the user"""

        if self.db_type == "weaviate":
            self.vector_db.add_memories( observation = semantic_memory)

        elif self.db_type == "pinecone":
            pass


    def _fetch_memories(self, observation: str,params) -> dict[str, str] | str:
        """Fetch related characteristics, preferences or dislikes for a user."""
        # self.init_pinecone(index_name=self.index)

        if self.db_type == "weaviate":

            return self.vector_db.fetch_memories(observation, params)

        elif self.db_type == "pinecone":
            pass


class LongTermMemory:
    def __init__(self, user_id: str = "676", memory_id:str=None, index_name: str = None, namespace:str=None, db_type:str="weaviate"):
        self.user_id = user_id
        self.memory_id = memory_id
        self.ltm_memory_id = str(uuid.uuid4())
        self.index_name = index_name
        self.namespace = namespace
        self.db_type = db_type
        # self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory(user_id = self.user_id, memory_id=self.memory_id, ltm_memory_id = self.ltm_memory_id, index_name=self.index_name, db_type=self.db_type)

class ShortTermMemory:
    def __init__(self, user_id: str = "676", memory_id:str=None, index_name: str = None, namespace:str=None, db_type:str="weaviate"):
        # Add any short-term memory-related attributes or setup here
        self.user_id = user_id
        self.memory_id = memory_id
        self.namespace = namespace
        self.db_type = db_type
        self.stm_memory_id = str(uuid.uuid4())
        self.index_name = index_name
        self.episodic_buffer = EpisodicBuffer(user_id=self.user_id, memory_id=self.memory_id, index_name=self.index_name, db_type=self.db_type)



class EpisodicBuffer:
    def __init__(self, user_id: str = "676", memory_id:str=None, index_name: str = None, namespace:str='EPISODICBUFFER', db_type:str="weaviate"):
        # Add any short-term memory-related attributes or setup here
        self.user_id = user_id
        self.memory_id = memory_id
        self.namespace = namespace
        self.db_type = db_type
        self.st_memory_id = "blah"
        self.index_name = index_name
        self.llm= ChatOpenAI(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="gpt-4-0613",
        )
        # self.vector_db = VectorDB(user_id=user_id, memory_id= self.memory_id, st_memory_id = self.st_memory_id, index_name=index_name, db_type=db_type, namespace=self.namespace)


        def _context_filter(self, context: str):
            """Filters the context for the buffer"""

            prompt = PromptTemplate.from_template(
                """ Based on the {CONTEXT} of {user_id} choose events that are relevant"""
            )

            return


        def _compute_weights(self, context: str):
            """Computes the weights for the buffer"""
            pass

        def _temporal_weighting(self, context: str):
            """Computes the temporal weighting for the buffer"""
            pass

    async def infer_schema_from_text(self, text: str):
        """Infer schema from text"""

        prompt_ = """ You are a json schema master. Create a JSON schema based on the following data and don't write anything else: {prompt} """

        complete_query = PromptTemplate(
            input_variables=["prompt"],
            template=prompt_,
        )

        chain = LLMChain(
            llm=self.llm, prompt=complete_query, verbose=True
        )
        chain_result = chain.run(prompt=text).strip()

        json_data = json.dumps(chain_result)
        return json_data

    def main_buffer(self, user_input=None):
        """AI function to convert unstructured data to structured data"""
        # Here we define the user prompt and the structure of the output we desire
        # prompt = output[0].page_content
        class PromptWrapper(BaseModel):
            observation: str = Field(
                description="observation we want to fetch from vectordb"
            )\
                # ,
            # json_schema: str = Field(description="json schema we want to infer")
        @tool("convert_to_structured", args_schema=PromptWrapper, return_direct=True)
        def convert_to_structured( observation=None, json_schema=None):
            """Convert unstructured data to structured data"""
            BASE_DIR = os.getcwd()
            json_path = os.path.join(BASE_DIR, "schema_registry", "ticket_schema.json")

            def load_json_or_infer_schema(file_path, document_path):
                """Load JSON schema from file or infer schema from text"""

                # Attempt to load the JSON file
                with open(file_path, 'r') as file:
                    json_schema = json.load(file)
                return json_schema

            json_schema =load_json_or_infer_schema(json_path, None)
            def run_open_ai_mapper(observation=None, json_schema=None):
                """Convert unstructured data to structured data"""

                prompt_msgs = [
                    SystemMessage(
                        content="You are a world class algorithm converting unstructured data into structured data."
                    ),
                    HumanMessage(content="Convert unstructured data to structured data:"),
                    HumanMessagePromptTemplate.from_template("{input}"),
                    HumanMessage(content="Tips: Make sure to answer in the correct format"),
                ]
                prompt_ = ChatPromptTemplate(messages=prompt_msgs)
                chain_funct = create_structured_output_chain(json_schema, prompt=prompt_, llm=self.llm, verbose=True)
                output = chain_funct.run(input=observation, llm=self.llm)
                yield output
            pipeline = dlt.pipeline(pipeline_name="train_ticket", destination='duckdb', dataset_name='train_ticket_data')
            info = pipeline.run(data=run_open_ai_mapper(prompt, json_schema))
            return print(info)


        class GoalWrapper(BaseModel):
            observation: str = Field(
                description="observation we want to fetch from vectordb"
            )

        @tool("fetch_memory_wrapper", args_schema=GoalWrapper, return_direct=True)
        def fetch_memory_wrapper(observation, args_schema=GoalWrapper):
            """Fetches data from the VectorDB and returns it as a python dictionary."""
            print("HELLO, HERE IS THE OBSERVATION: ", observation)

            marvin.settings.openai.api_key = os.environ.get('OPENAI_API_KEY')
            @ai_classifier
            class MemoryRoute(Enum):
                """Represents distinct routes  for  different memory types."""

                storage_of_documents_and_knowledge_to_memory = "SEMANTICMEMORY"
                raw_information_currently_processed_in_short_term_memory = "EPISODICBUFFER"
                raw_information_kept_in_short_term_memory = "SHORTTERMMEMORY"
                long_term_recollections_of_past_events_and_emotions = "EPISODICMEMORY"

            namespace= MemoryRoute(observation)
            vector_db = VectorDB(user_id=self.user_id, memory_id=self.memory_id, st_memory_id=self.st_memory_id,
                                 index_name=self.index_name, db_type=self.db_type, namespace=namespace.value)


            query = vector_db.fetch_memories(observation)

            return query

        class UpdatePreferences(BaseModel):
            observation: str = Field(
                description="observation we want to fetch from vectordb"
            )

        @tool("add_memories_wrapper", args_schema=UpdatePreferences, return_direct=True)
        def add_memories_wrapper(observation, args_schema=UpdatePreferences):
            """Updates user preferences in the VectorDB."""
            @ai_classifier
            class MemoryRoute(Enum):
                """Represents distinct routes  for  different memory types."""

                storage_of_documents_and_knowledge_to_memory = "SEMANTICMEMORY"
                raw_information_currently_processed_in_short_term_memory = "EPISODICBUFFER"
                raw_information_kept_in_short_term_memory = "SHORTTERMMEMORY"
                long_term_recollections_of_past_events_and_emotions = "EPISODICMEMORY"

            namespace= MemoryRoute(observation)
            print("HELLO, HERE IS THE OBSERVATION 2: ")
            vector_db = VectorDB(user_id=self.user_id, memory_id=self.memory_id, st_memory_id=self.st_memory_id,
                                 index_name=self.index_name, db_type=self.db_type, namespace=namespace.value)
            return vector_db.add_memories(observation)

        agent = initialize_agent(
            llm=self.llm,
            tools=[convert_to_structured,fetch_memory_wrapper, add_memories_wrapper],
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
        )

        prompt = """

            Based on all the history and information of this user, decide based on user query query: {query} which of the following tasks needs to be done:
            1. Memory retrieval , 2. Memory update,  3. Convert data to structured   If the query is not any of these, then classify it as 'Other'
            Return the result in format:  'Result_type': 'Goal', "Original_query": "Original query"
            """

        # template = Template(prompt)
        # output = template.render(query=user_input)
        # complete_query = output
        complete_query = PromptTemplate(
            input_variables=["query"], template=prompt
        )
        summary_chain = LLMChain(
            llm=self.llm, prompt=complete_query, verbose=True
        )
        from langchain.chains import SimpleSequentialChain

        overall_chain = SimpleSequentialChain(
            chains=[summary_chain, agent], verbose=True
        )
        output = overall_chain.run(user_input)
        return output




#DEFINE STM
#DEFINE LTM

class Memory:
    load_dotenv()

    def __init__(self, user_id: str = "676", index_name: str = None, knowledge_source: str = None,
                 knowledge_type: str = None, db_type:str="weaviate", namespace:str=None) -> None:
        self.user_id = user_id
        self.index_name = index_name
        self.db_type = db_type
        self.knowledge_source = knowledge_source
        self.knowledge_type = knowledge_type
        self.memory_id = str(uuid.uuid4())
        self.long_term_memory = LongTermMemory(user_id=self.user_id, memory_id=self.memory_id, index_name=index_name,
                                                namespace=namespace, db_type=self.db_type)
        self.short_term_memory = ShortTermMemory(user_id=self.user_id, memory_id=self.memory_id, index_name=index_name, db_type=self.db_type)


    def _update_semantic_memory(self, semantic_memory:str):
        return self.long_term_memory.semantic_memory._update_memories(
            memory_id=self.memory_id,
            semantic_memory=semantic_memory

        )

    def _fetch_semantic_memory(self, observation, params):
        return self.long_term_memory.semantic_memory._fetch_memories(
            observation=observation, params=params



        )

    def _run_buffer(self, user_input:str):
        return self.short_term_memory.episodic_buffer.main_buffer(user_input=user_input)

if __name__ == "__main__":
    namespace = "gggg"
    agent = Memory(index_name="my-agent", user_id='555' )
    #bb = agent._update_semantic_memory(semantic_memory="Users core summary")
    # bb = agent._fetch_semantic_memory(observation= "Users core summary", params =    {
    #     "path": ["inserted_at"],
    #     "operator": "Equal",
    #     "valueText": "*2023*"
    # })
    buffer = agent._run_buffer(user_input="I want to get a schema for my data")
    # print(bb)
    # rrr = {
    #     "path": ["year"],
    #     "operator": "Equal",
    #     "valueText": "2017*"
    # }

