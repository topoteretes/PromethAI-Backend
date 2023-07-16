import sys
from typing import Optional

sys.path.append('../llm_chains')
# from embedchain import EmbedChain

from llm_chains.chains import Agent
from embedchain import App
from llm_chains import AbstractAgent

from fastapi import FastAPI
# from sqlalchemy import create_engine, Column, Integer, String
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base
from embedchain.config import InitConfig, AddConfig, QueryConfig
from chromadb.utils import embedding_functions
import os
# Create the database engine
# engine = create_engine('postgresql://master:supersecreto@localhost:65432/agi_db')
# chatbots = {}
# # Create a session to interact with the database
# Session = sessionmaker(bind=engine)
# session = Session()

# Define the ORM base
# Base = declarative_base()

# Define the Chatbot model
# class Chatbot(Base):
#     __tablename__ = 'chatbots'
#     id = Column(Integer, primary_key=True)
#     user_id = Column(String(255))

# Create all defined tables in the database
# Base.metadata.create_all(bind=engine)


class AppAgent(App, Agent):
    def __init__(self, db="ChromaDB", table_name=None, user_id: Optional[str] = "676",
                 session_id: Optional[str] = None):
        App.__init__(self, InitConfig( ef=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            ), host="chromadb", port="8000", log_level="DEBUG"))
        Agent.__init__(self, table_name, user_id, session_id)

        self.app = App(InitConfig( ef=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            ), host="chromadb", port="8001", log_level="DEBUG"))

    def manage_resources(self, operation, resource_type=None, resource=None):
        """ Perform the specified operation (e.g., add, update, delete) on the given resource type."""
        if operation == "add":
            self.app.add(resource_type, resource)
        # elif operation == "update":
        #     self.update(resource_type, resource)
        elif operation == "delete":
            self.app.reset()
        else:
            raise ValueError(f"Invalid operation {operation}. Must be 'add', or 'delete'.")
        return

    def request_resources(self, operation, prompt=None):
        """ Perform the specified operation (e.g., query) on the given resource type."""
        if operation == "query":
            self.app.query(prompt)
        else:
            raise ValueError(f"Invalid operation {operation}. Must be 'query'.")
        return

    # def create_chatbot(self, user_id: str):
    #     App.add(Chatbot(user_id=user_id))
    #     session.commit()
    #     chatbots[user_id] = AppAgent(user_id=user_id)
    #
    # def delete_chatbot(self, user_id: str):
    #     if user_id in chatbots:
    #         session.query(Chatbot).filter_by(user_id=user_id).delete()
    #         session.commit()
    #         del chatbots[user_id]

# @app.post("/chatbot/{user_id}")
# async def create_chatbot_endpoint(user_id: str):
#     app_ch =AppAgent(user_id=user_id)
#     app_ch.create_chatbot(user_id)
#
# @app.delete("/chatbot/{user_id}")
# async def delete_chatbot_endpoint(user_id: str):
#     app_ch = AppAgent(user_id=user_id)
#     app.delete_chatbot(user_id)

# Other API endpoints for managing resources can be added similarly

# Example usage
# @app.post("/example")
# async def example():
#     user_id = "123"
#     category_name = "nutrition_chatbot"
#     resource_type = "pdf_file"
#     url = "https://navalmanack.s3.amazonaws.com/Eric-Jorgenson_The-Almanack-of-Naval-Ravikant_Final.pdf"
#     app_ch = AppAgent(user_id=user_id)
#     app_ch.create_chatbot(user_id)
#     chatbot = chatbots[user_id]
#     chatbot.manage_resources("add", resource_type, url)
#
#     return {"message": "Example executed successfully"}

# naval_chat_bot= AppAgent()
#
# naval_chat_bot.add("web_page", "https://nav.al/agi")
#
# # Embed Local Resources
# naval_chat_bot.add_local("qna_pair", (
# "Who is Naval Ravikant?", "Naval Ravikant is an Indian-American entrepreneur and investor."))
#
# naval_chat_bot.query(
#     "What unique capacity does Naval argue humans possess when it comes to understanding explanations or concepts?")

# Instantiate and run the AppAgent
if __name__ == "__main__":
    app_agent = AppAgent(user_id="1246")
    app_agent.manage_resources("add", "web_page", "https://nav.al/agi")