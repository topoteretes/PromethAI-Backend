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
from langchain import OpenAI, LLMMathChain

import os


import os
from datetime import datetime

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
    def __init__(self, user_id: str, index_name: str, db_type: str = "pinecone", weaviate_url: str = None):
        self.user_id = user_id
        self.index_name = index_name
        self.index = "my-agent"
        self.db_type = db_type
        self.weaviate_url = weaviate_url

        if self.db_type == "pinecone":
            self.vectorstore = self.init_pinecone(self.index_name)
        elif self.db_type == "weaviate":
            self.init_weaviate(self.index_name, weaviate_url)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def init_pinecone(self, index_name):
        load_dotenv()
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
        PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "")
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
        pinecone.Index(index_name)
        vectorstore: Pinecone = Pinecone.from_existing_index(

            index_name=self.index,
            embedding=OpenAIEmbeddings(),
            namespace='RESULT'
        )
        return vectorstore

    def init_weaviate(self, index_name, weaviate_url):
        auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get('WEAVIATE_API_KEY'))
        client = weaviate.Client(
            url=os.environ.get('WEAVIATE_INDEX'),
            auth_client_secret=auth_config,
            additional_headers={
                "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')
            }
        )
        # Initialize Weaviate here
        embeddings = OpenAIEmbeddings()
        # I use 'LangChain' for index_name and 'text' for text_key
        vectorstore = Weaviate(client, "LangChain", "text", embedding=embeddings)

    def update_memories(self, observation: str, namespace: str, page: str = "", source: str = ""):
        if self.db_type == "pinecone":
            # Update Pinecone memories here
            vectorstore: Pinecone = Pinecone.from_existing_index(
                index_name=self.index, embedding=OpenAIEmbeddings(), namespace=namespace
            )
            from datetime import datetime

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
                            "source": source,
                        },
                        namespace=namespace,
                    )
                ]
            )
        elif self.db_type == "weaviate":
            # Update Weaviate memories here

            vectorstore: Weaviate = weaviate.from_existing_index(
                index_name=self.index, embedding=OpenAIEmbeddings(), namespace=namespace
            )
            pass
    # def get_pinecone_vectorstore(self, namespace: str) -> pinecone.VectorStore:
    #     return Pinecone.from_existing_index(
    #         index_name=self.index, embedding=OpenAIEmbeddings(), namespace=namespace
    #     )

    def fetch_memories(self, observation: str, namespace: str):
        if self.db_type == "pinecone":
            # Fetch Pinecone memories here
            pass
        elif self.db_type == "weaviate":
            # Fetch Weaviate memories here
            pass


class ShortTermMemory:
    def __init__(self, user_id: str = "676", memory_id:str=None, index_name: str = None, knowledge_source:str=None):
        # Add any short-term memory-related attributes or setup here
        self.user_id = user_id
        self.memory_id = memory_id
        self.stm_memory_id = str(uuid.uuid4())
        self.knowledge_source = knowledge_source
        self.index_name = index_name
        self.index = "my-agent"
        self.episodic_buffer = EpisodicBuffer()
        self.actions = Actions(stm_memory_id=self.stm_memory_id)


class EpisodicBuffer:
    def __init__(self, user_id: str = "676", memory_id:str=None, index_name: str = None, knowledge_source:str=None):
        # Add any short-term memory-related attributes or setup here
        self.user_id = user_id
        self.memory_id = memory_id


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


        def _compute_buffer():
            """Computes buffer based on events, context and LTM"""
            prompt = PromptTemplate.from_template(
                "Based on the {EVENTS} and {CONTEXT} of {user_id} "
                + " choose action out of {ACTIONS} :\n"
                + "that can solve the problem observed in events and context"
                + "Do not embellish."
                + "\n\n Very short summary: "
            )
            # relevant_preferences = self._fetch_memories(
            #     f"Users core preferences", namespace="PREFERENCES"
            # )
            # relevant_dislikes = self._fetch_memories(
            #     f"Users core dislikes", namespace="PREFERENCES"
            # )
            # print(relevant_dislikes)
            # print(relevant_preferences)

            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)
            chain_results = chain.run(
                name=self.user_id,
                relevant_preferences=relevant_preferences,
                relevant_dislikes=relevant_dislikes,
            ).strip()
            print(chain_results)
            return chain_results

class Events:
    def __init__(self, user_id: str = "676", memory_id: str = None, index_name: str = None,
                 knowledge_source: str = None):
        # Add any short-term memory-related attributes or setup here
        self.user_id = user_id
        self.memory_id = memory_id

        def _update_events(self, memory_id: str = "None", semantic_memory: str = "None", namespace: str = "None",
                             source: str = "None") -> None:
            """Update event memory for the user"""
            vectorstore = self.vector_db.vectorstore
            retriever = vectorstore.as_retriever()
            metadata = {
                "inserted_at": datetime.now(),
                "memory_id": self.memory_id,
                "stm_memory_id": self.stm_memory_id,
                "event_id": str(uuid.uuid4()),
                "source": source,
                "last_updated_at": datetime.now(),
                "last_accessed_at": datetime.now()
            }

            retriever.add_documents(
                [
                    Document(
                        page_content=semantic_memory,
                        metadata=metadata,
                        namespace=namespace,
                    )
                ]
            )
    def _fetch_events(self, observation: str,knowledge_source:str, namespace: str) -> dict[str, str] | str:
        """Fetch related characteristics, preferences or dislikes for a user."""
        # self.init_pinecone(index_name=self.index)
        vectorstore = self.vector_db.vectorstore
        retriever = vectorstore.as_retriever()
        # retriever.search_kwargs = {"filter": {"user_id": {"$eq": self.user_id}}, {"knowledge_source": {"$eq": knowledge_type}}}
        answer_response = retriever.get_relevant_documents(observation)

        answer_response.sort(
            key=lambda doc: doc.metadata.get("inserted_at")
            if "inserted_at" in doc.metadata
            else datetime.min,
            reverse=True,
        )
        try:
            answer_response = answer_response[0]
        except IndexError:
            return {
                "error": "No document found for this user. Make sure that a query is appropriate"
            }
        return answer_response.page_content



class Actions:
    def __init__(self, user_id: str = "676", memory_id: str = None, index_name: str = None,
                 knowledge_source: str = None, stm_memory_id:str=None):
        # Add any short-term memory-related attributes or setup here
        self.user_id = user_id
        self.memory_id = memory_id
        self.vector_db = VectorDB(user_id=user_id, index_name=index_name, db_type="pinecone")
        self.stm_memory_id = stm_memory_id


    def _update_memories(self ,memory_id:str="None", semantic_memory: str="None", namespace: str="None", source: str = "None") -> None:
        """Update semantic memory for the user"""
        vectorstore = self.vector_db.vectorstore
        retriever = vectorstore.as_retriever()
        metadata = {
            "inserted_at": datetime.now(),
            "memory_id": self.memory_id,
            "stm_memory_id": self.stm_memory_id,
            "source": source,
            "last_updated_at": datetime.now(),
            "last_accessed_at": datetime.now()
        }

        retriever.add_documents(
            [
                Document(
                    page_content=semantic_memory,
                    metadata=metadata,
                    namespace=namespace,
                )
            ]
        )

    #episodic buffer
    #should contain the information from the current session

    # events
    # should contain tracking events
    # should also analyze episodic buffer and create events

    # actions
    # should have inject option for actions we want to add
    # should have actions inferred from the episodic buffer, and LTM

class EpisodicMemory:
    def __init__(self):
        # Add any episodic memory-related attributes or setup here
        pass

    # Add episodic memory methods here

class SemanticMemory:
    def __init__(self, user_id: str, memory_id:str, ltm_memory_id:str, index_name: str, knowledge_source:str):
        # Add any semantic memory-related attributes or setup here
        self.user_id=user_id
        self.index_name = index_name
        self.knowledge_source = knowledge_source
        self.namespace = "SEMANTICMEMORY"
        self.semantic_memory_id = str(uuid.uuid4())
        self.memory_id = memory_id
        self.ltm_memory_id = ltm_memory_id
        self.vector_db = VectorDB(user_id=user_id, index_name=index_name, db_type="pinecone")

    def _update_memories(self ,memory_id:str="None", semantic_memory: str="None", namespace: str="None", knowledge_source:str="None", document_type: str ="None", page:str="None", concept:str="None", source: str = "None") -> None:
        """Update semantic memory for the user"""
        vectorstore = self.vector_db.vectorstore
        retriever = vectorstore.as_retriever()
        metadata = {
            "inserted_at": datetime.now(),
            "memory_id": self.memory_id,
            "ltm_memory_id": self.ltm_memory_id,
            "semantic_memory_id": self.semantic_memory_id,
            "source": source,
            "knowledge_source": knowledge_source,
            "last_updated_at": datetime.now(),
            "last_accessed_at": datetime.now()
        }

        retriever.add_documents(
            [
                Document(
                    page_content=semantic_memory,
                    metadata=metadata,
                    namespace=namespace,
                )
            ]
        )

    def _fetch_memories(self, observation: str,knowledge_source:str, namespace: str) -> dict[str, str] | str:
        """Fetch related characteristics, preferences or dislikes for a user."""
        # self.init_pinecone(index_name=self.index)
        vectorstore = self.vector_db.vectorstore
        retriever = vectorstore.as_retriever()
        # retriever.search_kwargs = {"filter": {"user_id": {"$eq": self.user_id}}, {"knowledge_source": {"$eq": knowledge_type}}}
        answer_response = retriever.get_relevant_documents(observation)

        answer_response.sort(
            key=lambda doc: doc.metadata.get("inserted_at")
            if "inserted_at" in doc.metadata
            else datetime.min,
            reverse=True,
        )
        try:
            answer_response = answer_response[0]
        except IndexError:
            return {
                "error": "No document found for this user. Make sure that a query is appropriate"
            }
        return answer_response.page_content

class LongTermMemory:
    def __init__(self, user_id: str = "676", memory_id:str=None, index_name: str = None, knowledge_source:str=None):
        self.user_id = user_id
        self.memory_id = memory_id
        self.ltm_memory_id = str(uuid.uuid4())
        self.knowledge_source=knowledge_source
        self.index_name = index_name
        self.index = "my-agent"
        # self.init_pinecone(index_name)
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory(self.user_id, self.memory_id, self.ltm_memory_id, self.index_name, self.knowledge_type)






#DEFINE STM
#DEFINE LTM

class Memory:
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "")
    def __init__(self, user_id: str  = "676", index_name: str=None, knowledge_source:str=None) -> None:
        self.user_id = user_id
        self.index_name = index_name
        self.knowledge_source=knowledge_source
        self.memory = str(uuid.uuid4())
        self.index = "my-agent"
        self.vector_db = VectorDB(user_id=user_id, index_name=index_name)
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory(user_id=user_id,memory_id=self.memory, index_name=index_name, knowledge_type=self.knowledge_type)

        # add session id.
        # it should look for any memory with the current session id and extract memory id


    def _update_semantic_memory(self, semantic_memory, knowledge_source, document_type="", page=None, concept=None, source=""):
        return self.long_term_memory.semantic_memory._update_memories(
            memory_id=self.memory,
            semantic_memory=semantic_memory,
            namespace=self.long_term_memory.semantic_memory.namespace,
            knowledge_source=knowledge_source,
            document_type=document_type,
            page=page,
            concept=concept,
            source=source
        )

    def _fetch_semantic_memory(self, observation, knowledge_source):
        return self.long_term_memory.semantic_memory._fetch_memories(
            observation=observation,
            knowledge_source=knowledge_source,
            namespace=self.long_term_memory.semantic_memory.namespace
        )
    # def _update_memories(self, observation: str, namespace: str, page: str = "", source: str = "") -> None:
    #     """Update related characteristics, preferences or dislikes for a user."""
    #     vectorstore: Pinecone = Pinecone.from_existing_index(
    #         index_name=self.index_name, embedding=OpenAIEmbeddings(), namespace=namespace
    #     )
    #
    #     retriever = vectorstore.as_retriever()
    #     retriever.add_documents(
    #         [
    #             Document(
    #                 page_content=observation,
    #                 metadata={
    #                     "inserted_at": datetime.now(),
    #                     "text": observation,
    #                     "user_id": self.user_id,
    #                     "page": page,
    #                     "source": source,
    #                 },
    #                 namespace=namespace,
    #             )
    #         ]
    #     )
    #
    # def _fetch_memories(self, observation: str, namespace: str) -> dict[str, str] | str:
    #     """Fetch related characteristics, preferences or dislikes for a user."""
    #     self.init_pinecone(index_name=self.index)
    #     vectorstore: Pinecone = Pinecone.from_existing_index(
    #         index_name=self.index, embedding=OpenAIEmbeddings(), namespace=namespace
    #     )
    #     retriever = vectorstore.as_retriever()
    #     retriever.search_kwargs = {"filter": {"user_id": {"$eq": self.user_id}}}
    #     answer_response = retriever.get_relevant_documents(observation)
    #
    #     answer_response.sort(
    #         key=lambda doc: doc.metadata.get("inserted_at")
    #         if "inserted_at" in doc.metadata
    #         else datetime.min,
    #         reverse=True,
    #     )
    #     try:
    #         answer_response = answer_response[0]
    #     except IndexError:
    #         return {
    #             "error": "No document found for this user. Make sure that a query is appropriate"
    #         }
    #     return answer_response.page_content

if __name__ == "__main__":
    agent = Memory()
    bb = agent._update_semantic_memory(semantic_memory="Users core summary", knowledge_source="conceptual")
    #bb = agent._fetch_semantic_memory("Users core summary", knowledge_type="conceptual")
    print(bb)
