from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
import functools
import pinecone
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from pydantic import BaseModel, Field
import re
from jinja2 import Template
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain import LLMChain
from langchain.schema import  Document
from langchain.chains import SimpleSequentialChain
#from heuristic_experience_orchestrator.prompt_template_modification import PromptTemplate
# from langchain.retrievers import TimeWeightedVectorStoreRetriever
import os
from food_scrapers import wolt_tool
import json
from langchain.tools import GooglePlacesTool
import tiktoken
import asyncio
import logging
from langchain.chat_models import ChatOpenAI
# redis imports for cache

from langchain.cache import RedisSemanticCache
import langchain
import redis


import subprocess

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
from langchain.llms import Replicate

import os




class Agent():
    load_dotenv()
    OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4"
    GPLACES_API_KEY = os.getenv("GPLACES_API_KEY", "")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "")
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
    REDIS_HOST = os.getenv("REDIS_HOST", "promethai-dev-backend-redis-repl-gr.60qtmk.ng.0001.euw1.cache.amazonaws.com")

    def __init__(self, table_name=None, user_id: Optional[str] = "user123", session_id: Optional[str] = None) -> None:
        self.table_name = table_name
        self.user_id = user_id
        self.session_id = session_id
        self.memory = None
        self.thought_id_timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]   #  Timestamp with millisecond precision
        self.last_message = ""
        self.openai_model35 = "gpt-3.5-turbo"
        self.openai_model4 = "gpt-4"
        self.llm35 = ChatOpenAI(temperature=0.0,max_tokens = 2000, openai_api_key = self.OPENAI_API_KEY, model_name=self.openai_model35)
        self.llm = ChatOpenAI(temperature=0.0,max_tokens = 2000, openai_api_key = self.OPENAI_API_KEY, model_name="gpt-4")
        self.replicate_llm = Replicate(model="replicate/vicuna-13b:a68b84083b703ab3d5fbf31b6e25f16be2988e4c3e21fe79c2ff1c18b99e61c1", api_token=self.REPLICATE_API_TOKEN)
        self.verbose: bool = True

        self.openai_temperature = 0.0
        self.index = "my-agent"

        # from langchain.cache import RedisCache
        # from redis import Redis
        # langchain.llm_cache = RedisCache(redis_=Redis(host=self.REDIS_HOST, port=6379, db=0))

    def post_execution(func_to_execute):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                print("tu smo")
                result = func(self, *args, **kwargs)

                # extracting the model_speed argument from kwargs
                model_speed = kwargs.get('model_speed', 'slow')
                print("tu smo")
                # call the func_to_execute function
                getattr(self, func_to_execute.__name__)(model_speed)

                return result

            return wrapper

        return decorator
    def set_user_session(self, user_id: str, session_id: str) -> None:
        self.user_id = user_id
        self.session_id = session_id

    def get_ada_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002",api_key =OPENAI_API_KEY)[
            "data"
        ][0]["embedding"]

    def init_pinecone(self, index_name):
            load_dotenv()
            PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
            PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "")
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
            return pinecone.Index(index_name)



    # create the length function
    def tiktoken_len(self, text):
        tokenizer = tiktoken.get_encoding('cl100k_base')
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
    # class VectorDBInput(BaseModel):
    #     observation: str = Field(description="should be what we are inserting into the memory")
    #     namespace: str = Field(description="should be the namespace of the VectorDB")
    # @tool("_update_memories", return_direct=True, args_schema = VectorDBInput)
    def _update_memories(self, observation: str, namespace: str)-> None:
        """Update related characteristics, preferences or dislikes for a user."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        self.init_pinecone(index_name=self.index)
        vectorstore: Pinecone = Pinecone.from_existing_index(
            index_name=self.index,
            embedding=OpenAIEmbeddings(),
            namespace=namespace
        )
        from datetime import datetime
        retriever = vectorstore.as_retriever()
        retriever.add_documents([Document(page_content=observation,
                                          metadata={'inserted_at': datetime.now(), "text": observation,
                                                    'user_id': self.user_id}, namespace=namespace)])

    class FetchMemories(BaseModel):
        observation: str = Field(description="observation we want to fetch from vectordb")
        # namespace: str = Field(description="namespace of the VectorDB")
    # @tool("_update_memories", return_direct=True, args_schema=VectorDBInput)

    # @tool("_fetch_memories", args_schema = FetchMemories)
    def _fetch_memories(self, observation: str, namespace:str) -> List[Document]:
        """Fetch related characteristics, preferences or dislikes for a user."""

        self.init_pinecone(index_name=self.index)
        vectorstore: Pinecone = Pinecone.from_existing_index(
            index_name=self.index,
            embedding=OpenAIEmbeddings(),
            namespace=namespace
        )
        retriever = vectorstore.as_retriever()
        retriever.search_kwargs = {'filter': {'user_id': {'$eq': self.user_id}}}
        answer_response= retriever.get_relevant_documents(observation)

        answer_response.sort(key=lambda doc: doc.metadata.get('inserted_at') if 'inserted_at' in doc.metadata else datetime.min,
            reverse=True)
        try:
            answer_response = answer_response[0]
        except IndexError:
            return {"error": "No document found for this user. Make sure that a query is appropriate"}
        return answer_response


    def _compute_agent_summary(self, model_speed:str):
        """Computes summary for a person"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_preferences}"
            + "{relevant_dislikes}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        print("Computing Agent Summary")
        self.init_pinecone(index_name=self.index)
        # The agent seeks to think about their core characteristics.

        relevant_preferences = self._fetch_memories(f"Users core preferences", namespace="PREFERENCES")
        relevant_dislikes = self._fetch_memories(f"Users core dislikes", namespace="PREFERENCES")
        print(relevant_dislikes)
        print(relevant_preferences)

        if model_speed =='fast':
            output = self.replicate_llm(prompt)
            return output

        else:
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
            return chain.run(name= self.user_id, relevant_preferences=relevant_preferences.page_content, relevant_dislikes=relevant_dislikes.page_content).strip()



    def update_agent_preferences(self, preferences:str):
        """Serves to update agents preferences so that they can be used in summary"""

        prompt = """ The {name} has following {past_preference} and the new {preferences}
                Update user preferences and return a list of preferences
            Do not embellish.
            Summary: """
        self.init_pinecone(index_name=self.index)
        past_preference = self._fetch_memories(f"Users core preferences", namespace="PREFERENCE")
        prompt = PromptTemplate(input_variables=["name", "past_preference", "preferences"], template=prompt)

        # prompt = prompt.format(name=self.user_id, past_preference= past_preference, preferences=preferences)
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        chain_result = chain.run(prompt=prompt,  past_preference= past_preference, preferences=preferences, name=self.user_id).strip()
        return self._update_memories(chain_result, namespace="PREFERENCES")


    def update_agent_taboos(self, dislikes:str):
        """Serves to update agents taboos so that they can be used in summary"""
        prompt =""" The {name} has following {past_dislikes} and the new {dislikes}
                Update user taboos and return a list of dislikes
            Do not embellish.
            Summary: """
        self.init_pinecone(index_name=self.index)
        past_dislikes = self._fetch_memories(f"Users core dislikes", namespace="PREFERENCES")
        prompt = PromptTemplate(input_variables=["name", "past_dislikes", "dislikes"], template=prompt)
        # prompt = prompt.format(name=self.user_id, past_dislikes= past_dislikes, dislikes=dislikes)
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        chain_result = chain.run(prompt=prompt, name=self.user_id, past_dislikes= past_dislikes, dislikes=dislikes).strip()
        return self._update_memories(chain_result, namespace="PREFERENCES")
# ========
#         past_dislikes = self._fetch_memories(f"Users core dislikes", namespace="PREFERENCE")
#         prompt = PromptTemplate(input_variables=["name", "past_dislikes", "dislikes"], template=prompt)
#         prompt = prompt.format(name=self.user_id, past_dislikes= past_dislikes, dislikes=dislikes)
#         return self._update_memories(prompt, namespace="PREFERENCE")
# >>>>>>>> ffaa8fae584048339358bf02769341c5e4b70ef9:main_chains.py


    def update_agent_traits(self, traits:str):
        """Serves to update agent traits so that they can be used in summary"""
        prompt =""" The {name} has following {past_traits} and the new {traits}
                Update user traits and return a list of traits
            Do not embellish.
            Summary: """
        self.init_pinecone(index_name=self.index)

        past_traits = self._fetch_memories(f"Users core dislikes", namespace="PREFERENCES")
        prompt = PromptTemplate(input_variables=["name", "past_traits", "traits"], template=prompt)
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        chain_result = chain.run(prompt=prompt, past_traits= past_traits, traits=traits,name=self.user_id).strip()
        return self._update_memories(chain_result, namespace="PREFERENCES")
# ========
#         past_traits = self._fetch_memories(f"Users core dislikes", namespace="PREFERENCE")
#         prompt = PromptTemplate(input_variables=["name", "past_traits", "traits"], template=prompt)
#         prompt = prompt.format(name=self.user_id, past_traits= past_traits, traits=traits)
#         return self._update_memories(prompt, namespace="PREFERENCE")
# >>>>>>>> ffaa8fae584048339358bf02769341c5e4b70ef9:main_chains.py


    def update_agent_summary(self, model_speed):
        """Serves to update agent traits so that they can be used in summary"""
        summary = self._compute_agent_summary(model_speed=model_speed)
        return self._update_memories(summary, namespace="SUMMARY")

    def task_identification(self, goals:str):
        """Serves to update agent traits so that they can be used in summary"""
        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        complete_query = str(agent_summary) + goals
        complete_query = PromptTemplate.from_template(complete_query)
        print("HERE IS THE COMPLETE QUERY", complete_query)
        from heuristic_experience_orchestrator.task_identification import TaskIdentificationChain
        chain = TaskIdentificationChain.from_llm(llm=self.llm,  value="Decomposition", verbose=self.verbose)

        chain_output = chain.run(name= self.user_id).strip()
        return chain_output

    @post_execution(_compute_agent_summary)
    def solution_generation(self, factors:dict, model_speed:str):
        """Generates a solution choice"""
        import time

        start_time = time.time()
        prompt = """
                Help me choose what food choice, order, restaurant or a recipe to eat or make for my next meal.     
                There are {% for factor, value in factors.items() %}'{{ factor }}'{% if not loop.last %}, {% endif %}{% endfor %} factors I want to consider.
                {% for factor, value in factors.items() %}
                For '{{ factor }}', I want the meal to be '{{ value }}' points on a scale of 1 to 100 points{% if not loop.last %}.{% else %}.{% endif %}
                {% endfor %}
                Instructions and ingredients should be detailed.  Result type can be Recipe, but not Meal
                Answer with a result in a correct  python dictionary that is properly formatted that contains the following keys and must have  values
                "Result type" should be "Solution proposal,  "body" which should contain "proposal" and the value of the proposal that should be order, restaurant or a recipe
        """
        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(factors=factors)
        complete_query = str(agent_summary) + output
        # complete_query =  output
        complete_query = PromptTemplate.from_template(complete_query)

        if model_speed =='fast':
            output = self.replicate_llm(output)
            json_data = json.dumps(output)
            return json_data
        else:
            chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
            chain_result = chain.run(prompt=complete_query, name=self.user_id).strip()
            vectorstore: Pinecone = Pinecone.from_existing_index(
                index_name=self.index,
                embedding=OpenAIEmbeddings(),
                namespace='GOAL'
            )
            from datetime import datetime
            retriever = vectorstore.as_retriever()
            retriever.add_documents([Document(page_content=chain_result,
                                              metadata={'inserted_at': datetime.now(), "text": chain_result,
                                                        'user_id': self.user_id}, namespace="GOAL")])
            end_time = time.time()

            execution_time = end_time - start_time
            print("Execution time: ", execution_time, " seconds")
            json_data = json.dumps(chain_result)
            return json_data
    def recipe_generation(self,  prompt:str, model_speed:str):
        """Generates a recipe solution in json"""
        prompt_base = """ Help me choose what recipe to eat or make for my next meal based on this prompt {{prompt}}.     
                Instructions and ingredients should be detailed.
                 Answer a condensed JSON with no whitespaces that contains the following keys and values for every recipe in the list of field "recipes":
                 "title", "rating", "prep_time", "cook_time", "description", "ingredients", "instructions".  After the JSON output, dont explain or write anything
        """
        self.init_pinecone(index_name=self.index)
        # agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt_base)
        output = template.render(prompt=prompt)

        logging.info("HERE IS THE PROMPT", output)
        complete_query = output
        complete_query = PromptTemplate.from_template(complete_query)

        if model_speed =='fast':
            output = self.replicate_llm(output)
            return output
        else:
            logging.info("we are here")
            chain = LLMChain(llm=self.llm35, prompt=complete_query, verbose=self.verbose)
            chain_result = chain.run(prompt=complete_query, name=self.user_id).strip()
            #
            # vectorstore: Pinecone = Pinecone.from_existing_index(
            #     index_name=self.index,
            #     embedding=OpenAIEmbeddings(),
            #     namespace='RESULT'
            # )
            # from datetime import datetime
            # retriever = vectorstore.as_retriever()
            # retriever.add_documents([Document(page_content=chain_result,
            #                                   metadata={'inserted_at': datetime.now(), "text": chain_result,
            #                                             'user_id': self.user_id}, namespace="RESULT")])
            logging.info("HERE IS THE CHAIN RESULT", chain_result)
            return chain_result


    def prompt_to_choose_meal_tree(self, prompt: str, model_speed:str):
        """Serves to generate agent goals and subgoals based on a prompt"""

        json_example = {"prompt":prompt,"tree":[{"category":"time","options":[{"category":"quick","options":[{"category":"1 min"},{"category":"10 mins"},{"category":"30 mins"}],"preference":[]},{"category":"slow","options":[{"category":"60 mins"},{"category":"120 mins"},{"category":"180 mins"}],"preference":[]}],"preference":["quick"]}]}
        json_str = str(json_example)
        json_str = json_str.replace("{", "{{").replace("}", "}}")
        prompt_template=""" Decompose {{ prompt_str }} statement into four decision points that are 
        relevant to statement above, personal to the user if possible and that he should apply to optimize his decision choices related to food.
         Also, help me decompose the decision points into five categories each, starting with the default option provided in the statement. 
         For each of the four options  provide a mind map representation of the four secondary nodes that can be used to narrow down the choice better. Don't leave options blank.
         Please provide the response in JSON format with proper syntax, ensuring that all strings are enclosed in double quotes,  in maximum three lines with no whitespaces. The structure should follow this structure : {{json_str}}
        """

        self.init_pinecone(index_name=self.index)
        # agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt_template)
        output = template.render(prompt_str = prompt, json_str=json_str)
        complete_query =  output
        print("HERE IS THE COMPLETE QUERY", complete_query)
        complete_query = PromptTemplate.from_template(complete_query)

        if model_speed =='fast':
            output = self.replicate_llm(output)
            json_data = json.dumps(output)
            return json_data
        else:
            chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
            chain_result = chain.run(prompt=complete_query, name=self.user_id).strip()
            print("HERE IS THE CHAIN RESULT", chain_result)
            vectorstore: Pinecone = Pinecone.from_existing_index(
                index_name=self.index,
                embedding=OpenAIEmbeddings(),
                namespace='GOAL'
            )
            from datetime import datetime
            retriever = vectorstore.as_retriever()
            logging.info(str(chain_result))
            retriever.add_documents([Document(page_content=chain_result,
                                            metadata={'inserted_at': datetime.now(), "text": chain_result,
                                                        'user_id': self.user_id}, namespace="GOAL")])
            # chain_result=str(chain_result)
            chain_result = json.dumps(chain_result)
            return chain_result

    def prompt_to_update_meal_tree(self,  category:str, from_:str, to_:str, model_speed:str):

        self.init_pinecone(index_name=self.index)
        vectorstore: Pinecone = Pinecone.from_existing_index(
            index_name=self.index,
            embedding=OpenAIEmbeddings(),
            namespace="GOAL"
        )

        retriever = vectorstore.as_retriever()
        retriever.search_kwargs = {'filter': {'user_id': {'$eq': self.user_id}}}  # filter by user_id
        answer_response = retriever.get_relevant_documents("prompt")
        answer_response.sort(
            key=lambda doc: doc.metadata.get('inserted_at') if 'inserted_at' in doc.metadata else datetime.min,
            reverse=True)
        logging.info(str(answer_response))
        # The most recent document is now the first element of the list.
        try:
            most_recent_document = answer_response[0]
        except IndexError:
            return {"error": "No document found for this user. Make sure that a query is appropriate"}
        escaped_content = most_recent_document.page_content.replace("{", "{{").replace("}", "}}")



        logging.info(escaped_content)
        # print("HERE IS THE ESCAPED CONTENT", escaped_content)
        # change category that exists in the tree
        # add new category that doesn't exist in the tree
        optimization_prompt = """Change the category: {{category}} based on {{from_}} to {{to_}}  change and update appropriate of the following original: {{results}}
         """

        optimization_prompt = Template(optimization_prompt)
        optimization_output = optimization_prompt.render(category=category, from_=from_, to_=to_,  results=escaped_content)
        complete_query = PromptTemplate.from_template(optimization_output)
        # prompt_template = PromptTemplate(input_variables=["query"], template=optimization_output)
        review_chain = LLMChain(llm=self.llm35, prompt=complete_query)
        review_chain_result = review_chain.run(prompt=complete_query, name=self.user_id).strip()
        print("HERE IS THE OUTPUT", review_chain_result)
        json_data = json.dumps(review_chain_result)
        return json_data

     # def goal_generation(self, factors: dict, model_speed:str):
     #
     #     """Serves to optimize agent goals"""
     #
     #     prompt = """
     #          Based on all the history and information of this user, suggest mind map that would have four decision points that are personal to him that he should apply to optimize his decision choices related to food. It must be food and nutrition related.
     #          The cuisine should be one of the points, and goal should contain one or maximum three words. If user preferences don't override it, and if it includes time elemeent, it should be called "Time to make", if it descibes complexity of the dish, it should be  "Complexity", and if descibing food content it should be called "Macros"
     #          Answer a condensed JSON with no whitespaces. The structure should only contain a list of goals under field "goals". After the JSON output, don't explain or write anything.
     #        """
     #
     #     self.init_pinecone(index_name=self.index)
     #     agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
     #     template = Template(prompt)
     #     output = template.render(factors=factors)
     #     complete_query = str(agent_summary) + output
     #     complete_query = PromptTemplate.from_template(complete_query)
     #     if model_speed =='fast':
     #        output = self.replicate_llm(output)
     #        return output
     #     else:
     #        chain = LLMChain(llm=self.llm,  prompt=complete_query, verbose=self.verbose)
     #        chain_result = chain.run(prompt=complete_query).strip()
     #        vectorstore: Pinecone = Pinecone.from_existing_index(
     #            index_name=self.index,
     #            embedding=OpenAIEmbeddings(),
     #            namespace='GOAL'
     #        )
     #        from datetime import datetime
     #        retriever = vectorstore.as_retriever()
     #        retriever.add_documents([Document(page_content=chain_result,
     #                                          metadata={'inserted_at': datetime.now(), "text": chain_result,
     #                                                    'user_id': self.user_id}, namespace="GOAL")])
     #        return chain_result

    # def sub_goal_generation(self, factors: dict, model_speed:str):
    #     """Serves to generate sub goals for the user and drill down into it"""
    #
    #     prompt = """
    #         Base
    #         d on all the history and information of this user, GOALS PROVIDED HERE  {% for factor in factors %} '{{ factor['name'] }}'{% if not loop.last %}, {% endif %}{% endfor %}
    #          provide a mind map representation of the secondary nodes that can be used to narrow down the choice better.It needs to be food and nutrition related. Each of the results should have 4 sub nodes.
    #         Answer a condensed JSON with no whitespaces. The strucuture should only contain the list of subgoal items under field "sub_goals".
    #         Every subgoal should have a "goal_name" refers to the goal and the list of subgoals with "name" and a "amount" should be shown as a range from 0 to 100, with a value chosen explicilty and shown based on the personal preferences of the user.
    #         After the JSON output, don't explain or write anything
    #         """
    #
    #     self.init_pinecone(index_name=self.index)
    #     agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
    #     template = Template(prompt)
    #     output = template.render(factors=factors)
    #     print("HERE IS THE AGENT SUMMARY", agent_summary)
    #     print("HERE IS THE TEMPLATE", output)
    #     complete_query = str(agent_summary) + output
    #     complete_query = PromptTemplate.from_template(complete_query)
    #     if model_speed =='fast':
    #         output = self.replicate_llm(output)
    #         return output
    #     else:
    #         chain = LLMChain(llm=self.llm,  prompt=complete_query, verbose=self.verbose)
    #         chain_result = chain.run( prompt=complete_query).strip()
    #         vectorstore: Pinecone = Pinecone.from_existing_index(
    #             index_name=self.index,
    #             embedding=OpenAIEmbeddings(),
    #             namespace='SUBGOAL'
    #         )
    #         from datetime import datetime
    #         retriever = vectorstore.as_retriever()
    #         retriever.add_documents([Document(page_content=chain_result,
    #                                           metadata={'inserted_at': datetime.now(), "text": chain_result,
    #                                                     'user_id': self.user_id}, namespace="SUBGOAL")])
    #         print("RESULT IS ", chain_result)
    #         return chain_result

    def extract_info(self, s):
        lines = s.split('\n')
        name = lines[0]
        address = lines[1].replace('Address: ', '')
        phone = lines[2].replace('Phone: ', '')
        website = lines[3].replace('Website: ', '')
        return {
            'name': name,
            'address': address,
            'phone': phone,
            'website': website,
        }
    async def restaurant_generation(self, prompt: str, model_speed:str):
        """Serves to suggest a restaurant to the agent"""

        prompt = """
              Based on the following prompt {{prompt}} and all the history and information of this user,
                Determine the type of restaurant you should offer to a customer. Make the recomendation very short and to a point, as if it is something you would type on google maps
            """

        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(prompt=prompt)
        complete_query = str(agent_summary) + output
        complete_query = PromptTemplate.from_template(complete_query)
        chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
        chain_result = chain.run(prompt=complete_query).strip()
        GPLACES_API_KEY = self.GPLACES_API_KEY
        places = GooglePlacesTool()
        output = places.run(chain_result)
        restaurants = re.split(r'\d+\.', output)[1:3]
        # Create a list of dictionaries for each restaurant
        restaurant_list = [self.extract_info(r) for r in restaurants]
        print('HERE IS THE OUTPUT', restaurant_list)
        return restaurant_list
    async def run_wolt_tool(self, zipcode, chain_result):
        from food_scrapers import  wolt_tool
        return wolt_tool.main(zipcode, chain_result)
    async def delivery_generation(self, prompt: str, zipcode:str, model_speed:str):
        """Serves to optimize agent delivery recommendations"""

        prompt = """
              Based on the following prompt {{prompt}}
                Determine the type of food you would want to recommend to the user, that is commonly ordered online. It should of type of food offered on a delivery app similar to burger or pizza, but it doesn't have to be that. 
                The response should be very short
            """

        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(prompt=prompt)
        complete_query = str(agent_summary) + output
        complete_query = PromptTemplate.from_template(complete_query)
        chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
        chain_result = chain.run(prompt=complete_query).strip()
        from food_scrapers import wolt_tool
        output = await wolt_tool.main( zipcode=zipcode, prompt=chain_result)
        return output




    def voice_input(self, query: str, model_speed:str):
        """Serves to generate sub goals for the user and drill down into it"""

        prompt = """ 
        {bu}
            Based on all the history and information of this user, classify the following query: {{query}} into one of the following categories:
            1. Goal update , 2. Preference change,  3. Result change 4. Subgoal update  If the query is not any of these, then classify it as 'Other'
            Return the classification and a very short summary of the query as a python dictionary. Update or replace or remove the original factors with the new factors if it is specified.
            with following python dictionary format 'Result_type': 'Goal', "Result_action": "Goal changed", "value": "Diet added", "summary": "The user is updating their goal to lose weight"
            Make sure to include the factors in the summary if they are provided
            """

        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(query=query)
        print("HERE IS THE AGENT SUMMARY", agent_summary)
        print("HERE IS THE TEMPLATE", output)
        complete_query =  output
        complete_query = PromptTemplate(input_variables=["bu"], template=complete_query)
        if model_speed =='fast':
            output = self.replicate_llm(output)
            return output
        else:
            chain = LLMChain(llm=self.llm,  prompt=complete_query, verbose=self.verbose)
            summary_action = chain.run("bu").strip()
            summary_action = summary_action.split("Result_type")[1].split("summary")[0].strip()
            summary_action = summary_action.split(":")[1].strip()
            print(summary_action)
            if 'goal' in summary_action.lower():
                namespace_val= "GOAL"
            elif 'preference' in summary_action.lower():
                namespace_val = "PREFERENCE"
            elif 'result' in summary_action.lower():
                namespace_val = "RESULT"
            elif 'subgoal' in summary_action.lower():
                namespace_val = "GOAL"
            else:
                namespace_val = "OTHER"
            vectorstore: Pinecone = Pinecone.from_existing_index(
                index_name=self.index,
                embedding=OpenAIEmbeddings(),
                namespace=namespace_val
            )

            retriever = vectorstore.as_retriever()
            retriever.search_kwargs = {'filter': {'user_id': {'$eq': self.user_id}}} # filter by user_id
            answer_response = retriever.get_relevant_documents("prompt")
            logging.info(answer_response)
            answer_response.sort(key=lambda doc: doc.metadata.get('inserted_at') if 'inserted_at' in doc.metadata else datetime.min, reverse=True)
            # The most recent document is now the first element of the list.
            try:
                most_recent_document = answer_response[0]
            except IndexError:
                return {"error": "No document found for this user. Make sure that a query is appropriate"}
            escaped_content = most_recent_document.page_content.replace("{", "{{").replace("}", "}}")

            optimization_prompt = """Based on the query: {query} change and update appropriate of the following original: {{results}}"""

            optimization_prompt = Template(optimization_prompt)
            optimization_output = optimization_prompt.render( results=escaped_content)
            prompt_template = PromptTemplate(input_variables=["query"], template=optimization_output)
            review_chain = LLMChain(llm=self.llm, prompt=prompt_template)
            output = review_chain.run(query=summary_action).strip()


            json_data = json.dumps(output)

            return json_data

            # overall_chain = SimpleSequentialChain(chains=[chain, review_chain], verbose=True)
            # final_output = overall_chain.run('bu').strip()
            # print("HERE IS THE FINAL OUTPUT", final_output)
            # json_data = json.dumps(final_output)
            # return json_data

    def voice_text_input_imp(self, query: str, model_speed: str):

        """Serves to generate sub goals for the user and drill down into it"""
        from langchain.agents import initialize_agent, AgentType
        from pydantic import BaseModel, Field
        from langchain import PromptTemplate
        import os
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
        from langchain.llms.openai import OpenAI
        from langchain.tools import BaseTool, StructuredTool, Tool, tool
        # llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        llm=self.llm

        class FetchMemories(BaseModel):
            observation: str = Field(description="observation we want to fetch from vectordb")
            # namespace: str = Field(description="namespace of the VectorDB")

        # @tool("_update_memories", return_direct=True, args_schema=VectorDBInput)

        @tool("memories_wrapper", args_schema=FetchMemories, return_direct=True)
        def memories_wrapper(observation,  args_schema=FetchMemories):
            """Fetches data from the VectorDB and returns it as a python dictionary."""
            return self._fetch_memories(observation, "GOAL")

        class OptimisticString(BaseModel):
            input_string: str = Field(description="should be a string with any tone")

        @tool("optimistic_string", args_schema=OptimisticString)
        def optimistic_string(input_string: str) -> str:
            """Rewrites the input string with a more optimistic tone."""
            # Add your logic to process the input_string and generate the output_string
            template = (
                "Rewrite the following sentence with a more optimistic tone: {input_string}"
            )
            prompt = PromptTemplate(template=template, input_variables=["input_string"])

            output_string = llm(
                prompt.format(input_string=input_string)
            )  # Replace this with the actual call to the language model
            return output_string
        class ActionChoice(BaseModel):
            input_string: str = Field(description="should be a string with any tone")

        @tool("action_choice", args_schema=ActionChoice, return_direct=True)
        def action_choice(summary_action: str) -> str:
            """Rewrites the input string with a more optimistic tone."""
            summary_action = summary_action.split("Result_type")[1].split("summary")[0].strip()
            summary_action = summary_action.split(":")[1].strip()
            print(summary_action)
            if 'goal' in summary_action.lower():
                namespace_val = "GOAL"
            elif 'preference' in summary_action.lower():
                namespace_val = "PREFERENCE"
            elif 'result' in summary_action.lower():
                namespace_val = "RESULT"
            elif 'subgoal' in summary_action.lower():
                namespace_val = "GOAL"
            else:
                namespace_val = "OTHER"
            return namespace_val
        # agent_instance = Agent()

        agent = initialize_agent(llm=llm, tools=[memories_wrapper, action_choice], agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        # result_most_recent_memory_doc = agent.run("Return json content from vectordb")
        prompt = """ 
        {bu}
            Based on all the history and information of this user, classify the following query: {{query}} into one of the following categories:
            1. Goal update , 2. Preference change,  3. Result change 4. Subgoal update  If the query is not any of these, then classify it as 'Other'
            Return the classification and a very short summary of the query as a python dictionary. Update or replace or remove the original factors with the new factors if it is specified.
            with following python dictionary format 'Result_type': 'Goal', "Result_action": "Goal changed", "value": "Diet added", "summary": "The user is updating their goal to lose weight"
            Make sure to include the factors in the summary if they are provided
            """

        # print(result)
        template = Template(prompt)
        output = template.render(query=query)
        complete_query =  output
        complete_query = PromptTemplate(input_variables=["bu"], template=complete_query)
        chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
        # summary_action = chain.run("bu").strip()
        # summary_action = summary_action.split("Result_type")[1].split("summary")[0].strip()
        # summary_action = summary_action.split(":")[1].strip()
        # print(summary_action)
        # if 'goal' in summary_action.lower():
        #     namespace_val = "GOAL"
        # elif 'preference' in summary_action.lower():
        #     namespace_val = "PREFERENCE"
        # elif 'result' in summary_action.lower():
        #     namespace_val = "RESULT"
        # elif 'subgoal' in summary_action.lower():
        #     namespace_val = "GOAL"
        # else:
        #     namespace_val = "OTHER"
        #

        # Find the index of the first opening curly bracket
        start_index = result_most_recent_memory_doc.find('{')
        # Find the index of the corresponding closing curly bracket
        end_index = result_most_recent_memory_doc.find('}', start_index) + 1

        # Extract the substring between the first opening and closing curly brackets
        result_most_recent_memory_doc = result_most_recent_memory_doc[start_index:end_index]

        escaped_content = result_most_recent_memory_doc.replace("{", "{{").replace("}", "}}")
        optimization_prompt = """Based on the query: {query} change and update appropriate of the following original: {{results}}"""
        optimization_prompt = Template(optimization_prompt)
        optimization_output = optimization_prompt.render( results=escaped_content)
        prompt_template = PromptTemplate(input_variables=["query"], template=optimization_output)
        review_chain = LLMChain(llm=self.llm, prompt=prompt_template)
        # output = review_chain.run(query=summary_action).strip()
        # overall_chain = SimpleSequentialChain(chains=[chain, review_chain], verbose=True)
        # final_output = overall_chain.run('bu').strip()
        json_data = json.dumps(output)
        return json_data
        # prompt = """
        # {bu}
        #     Based on all the history and information of this user, classify the following query: {{query}} into one of the following categories:
        #     1. Goal update , 2. Preference change,  3. Result change 4. Subgoal update  If the query is not any of these, then classify it as 'Other'
        #     Return the classification and a very short summary of the query as a python dictionary. Update or replace or remove the original factors with the new factors if it is specified.
        #     with following python dictionary format 'Result_type': 'Goal', "Result_action": "Goal changed", "value": "Diet added", "summary": "The user is updating their goal to lose weight"
        #     Make sure to include the factors in the summary if they are provided
        #     """
        # # from langchain.agents import load_tools
        # # tools = load_tools(["serpapi", "llm-math", "_fetch_memories", "_update_memories"], llm=self.llm)
        # from langchain import LLMMathChain, SerpAPIWrapper
        # from langchain.agents import AgentType, initialize_agent
        # from langchain.chat_models import ChatOpenAI
        # from langchain.tools import BaseTool, StructuredTool, Tool, tool
        # # search = SerpAPIWrapper()
        # # llm_math_chain = LLMMathChain(llm=self.llm, verbose=True)
        #
        # # class FetchMemories(BaseModel):
        # #     observation: str = Field(description="observation we want to fetch from vectordb")
        # #     namespace: str = Field(description="namespace of the VectorDB")
        # # tools = [
        # #     Tool.from_function(
        # #         func=search.run,
        # #         name="Search",
        # #         description="useful for when you need to answer questions about current events"
        # #         # coroutine= ... <- you can specify an async method if desired as well
        # #     ),
        # # ]
        # # tools.append(
        # #     Tool.from_function(
        # #         func=self._fetch_memories,
        # #         name="_fetch_memories",
        # #         description="useful for when you need to answer questions about math",
        # #         args_schema=FetchMemories
        # #         # coroutine= ... <- you can specify an async method if desired as well
        # #     )
        # # )
        # llm = OpenAI(temperature=0, openai_api_key = self.OPENAI_API_KEY)
        # class OptimisticString(BaseModel):
        #     input_string: str = Field(description="should be a string with any tone")
        # @tool("optimistic_string", args_schema=OptimisticString)
        # def optimistic_string(input_string: str) -> str:
        #     """Rewrites the input string with a more optimistic tone."""
        #     # Add your logic to process the input_string and generate the output_string
        #     template = (
        #         "Rewrite the following sentence with a more optimistic tone: {input_string}"
        #     )
        #     prompt = PromptTemplate(template=template, input_variables=["input_string"])
        #
        #     output_string = llm(
        #         prompt.format(input_string=input_string)
        #     )  # Replace this with the actual call to the language model
        #     return output_string
        # # _fetch_memories = load_tools(["_fetch_memories"], llm=self.llm)
        # from langchain.agents import initialize_agent
        # agento = initialize_agent(llm, tools=[optimistic_string],  verbose=True)
        # # tool_names = [tool.name for tool in tools]
        # # agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        # agento.run("users core summary", namespace="SUMMARY")
        # self.init_pinecone(index_name=self.index)
        # agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        # template = Template(prompt)
        # output = template.render(query=query)
        # print("HERE IS THE AGENT SUMMARY", agent_summary)
        # print("HERE IS THE TEMPLATE", output)
        # complete_query = output
        # complete_query = PromptTemplate(input_variables=["bu"], template=complete_query)
        # if model_speed == 'fast':
        #     output = self.replicate_llm(output)
        #     return output
        # else:
        #     chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
        #     summary_action = chain.run("bu").strip()
        #     summary_action = summary_action.split("Result_type")[1].split("summary")[0].strip()
        #     summary_action = summary_action.split(":")[1].strip()
        #     print(summary_action)
        #     if 'goal' in summary_action.lower():
        #         namespace_val = "GOAL"
        #     elif 'preference' in summary_action.lower():
        #         namespace_val = "PREFERENCE"
        #     elif 'result' in summary_action.lower():
        #         namespace_val = "RESULT"
        #     elif 'subgoal' in summary_action.lower():
        #         namespace_val = "GOAL"
        #     vectorstore: Pinecone = Pinecone.from_existing_index(
        #         index_name=self.index,
        #         embedding=OpenAIEmbeddings(),
        #         namespace=namespace_val
        #     )
        #
        #     retriever = vectorstore.as_retriever()
        #     retriever.search_kwargs = {'filter': {'user_id': {'$eq': self.user_id}}}  # filter by user_id
        #     print(retriever.get_relevant_documents(summary_action))
        #     answer_response = retriever.get_relevant_documents(summary_action)
        #     answer_response.sort(key=lambda doc: doc.metadata.get(
        #         'inserted_at') if 'inserted_at' in doc.metadata else datetime.min, reverse=True)
        #     # The most recent document is now the first element of the list.
        #     try:
        #         most_recent_document = answer_response[0]
        #     except IndexError:
        #         return {"error": "No document found for this user. Make sure that a query is appropriate"}
        #     escaped_content = most_recent_document.page_content.replace("{", "{{").replace("}", "}}")
        #     optimization_prompt = """Based on the query: {query} change and update appropriate of the following original: {{results}}"""
        #     optimization_prompt = Template(optimization_prompt)
        #     optimization_output = optimization_prompt.render(results=escaped_content)
        #     prompt_template = PromptTemplate(input_variables=["query"], template=optimization_output)
        #     review_chain = LLMChain(llm=self.llm, prompt=prompt_template)
        #     output = review_chain.run(query=summary_action).strip()
        #     json_data = json.dumps(output)
        #     return json_data

            # chain_result = chain.run(prompt=complete_query).strip()
            # json_data = json.dumps(chain_result)
            # return json_data
            # self._update_memories(observation="Man walks in a forest", namespace="test_namespace")
            # template = """
            # {summaries}
            # {question}
            # """

            # embeddings = OpenAIEmbeddings()
            # embeddings.embed_documents(["man walks in the forest", "horse walks in the field"])
            # from langchain.chains import RetrievalQAWithSourcesChain
            # from langchain.chains import RetrievalQA
            # llm = OpenAIEmbeddings()

            # from langchain.retrievers import TimeWeightedVectorStoreRetriever
            # retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.0000000000000000000000001,
            #                                              k=1)
            # yesterday = datetime.now() - timedelta(days=1)
            # #retriever.add_documents([Document(page_content="hello majmune", metadata={"last_accessed_at": yesterday,"text": "hello majmune",'user_id': self.user_id}, namespace="test_namespace")])
            # # retriever.add_documents([Document(page_content="hello foo", metadata={ "text": "hello foo"} ,namespace="test_namespace")])
            # print(retriever.get_relevant_documents("hello majmune"))
            # from langchain.chains.qa_with_sources import load_qa_with_sources_chain
            # qa_chain = load_qa_with_sources_chain(llm=self.llm, chain_type="stuff")

            #this works
            # from langchain.chains.question_answering import load_qa_chain
            # qa_chain = load_qa_chain(self.llm, chain_type="stuff")
            # qa = RetrievalQA(
            #     combine_documents_chain=qa_chain,
            #     retriever=retriever,
            #     verbose=True,
            # )
            # #
            # answer_response = qa.run("give me an answer in python dictionary format with the key of the result type and the value of the result action")
            #
            # print(answer_response)
            # from operator import itemgetter
            # answer_response.sort(key=lambda doc: doc.metadata['inserted_at'], reverse=True)
            #
            # # The most recent document is now the first element of the list.
            # most_recent_document = answer_response[0]
            # print(answer_response)
            #above works until works
    # def simple_agent_chain(self):
    #     """To test simple agent and use intermediary steps"""
    #     from langchain.agents import load_tools
    #     tools = load_tools(["serpapi", "llm-math"], llm=self.llm)
    #     agent = initialize_agent(tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
    #                              return_intermediate_steps=True)
    #     response = agent(
    #         {"input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"})
    #     print(response["intermediate_steps"])
    #     return
    def solution_evaluation_test(self):
        """Serves to update agent traits so that they can be used in summary"""
        return


    def solution_implementation(self):
        """Serves to update agent traits so that they can be used in summary"""
        return

    # checker_chain = LLMSummarizationCheckerChain(llm=llm, verbose=True, max_checks=2)
    # text = """
    # Your 9-year old might like these recent discoveries made by The James Webb Space Telescope (JWST):
    # • In 2023, The JWST spotted a number of galaxies nicknamed "green peas." They were given this name because they are small, round, and green, like peas.
    # • The telescope captured images of galaxies that are over 13 billion years old. This means that the light from these galaxies has been traveling for over 13 billion years to reach us.
    # • JWST took the very first pictures of a planet outside of our own solar system. These distant worlds are called "exoplanets." Exo means "from outside."
    # These discoveries can spark a child's imagination about the infinite wonders of the universe."""
    # checker_chain.run(text)
    def _retrieve_summary(self):
        """Serves to retrieve agent summary"""
        self.init_pinecone(index_name=self.index)
        result = self._fetch_memories('Users core prompt', "GOAL")
        print(result)
        return result




if __name__ == "__main__":
    agent = Agent()
    # agent.goal_optimization(factors={}, model_speed="slow")
    # agent._update_memories("lazy, stupid and hungry", "TRAITS")
    # agent.update_agent_traits("His personality is greedy")
    # agent.update_agent_preferences("Alergic to corn")
    # agent.update_agent_taboos("Dislike is brocolli")
    #agent.update_agent_summary(model_speed="slow")
    # agent.simple_agent_chain()
    #result = agent.prompt_to_choose_meal_tree(" I’d like a quick veggie meal under 25$ near me.", model_speed="slow")
    #print(result)
    # agent._test()
    # agent._retrieve_summary()
    agent.voice_text_input_imp("Users core prompt ", model_speed="slow")
    # agent.goal_generation( {    'health': 85,
    # 'time': 75,
    # 'cost': 50}, model_speed="slow")