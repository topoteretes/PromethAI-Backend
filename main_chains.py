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
import datetime
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
    OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
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
        self.thought_id_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]   #  Timestamp with millisecond precision
        self.last_message = ""
        self.llm = OpenAI(temperature=0.0,max_tokens = 1000, openai_api_key = self.OPENAI_API_KEY)
        self.replicate_llm = Replicate(model="replicate/vicuna-13b:a68b84083b703ab3d5fbf31b6e25f16be2988e4c3e21fe79c2ff1c18b99e61c1", api_token=self.REPLICATE_API_TOKEN)
        self.verbose: bool = True
        self.openai_model = "gpt-3.5-turbo"
        self.openai_temperature = 0.0
        self.index = "my-agent"

        from langchain.cache import RedisCache
        from redis import Redis
        langchain.llm_cache = RedisCache(redis_=Redis(host=self.REDIS_HOST, port=6379, db=0))

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
    class VectorDBInput(BaseModel):
        observation: str = Field(description="should be what we are inserting into the memory")
        namespace: str = Field(description="should be the namespace of the VectorDB")

    def _update_memories(self, observation: str, namespace: str):

        """Update related characteristics, preferences or dislikes for a user."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        memory = self.init_pinecone(index_name=self.index)


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=self.tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Split observation into chunks
        record_texts = text_splitter.split_text(observation)
        metadata = {
            'thought_string': observation,
            'user_id': self.user_id,
            'source': "user_input",
        }

        # Create individual metadata dicts for each chunk
        record_metadatas = [{
            'chunk': j,
            'text': text,
            **metadata
        } for j, text in enumerate(record_texts)]

        vectors = []

        # Get ada embedding for each chunk and create a vector
        for record in record_metadatas:
            vector = self.get_ada_embedding(record['text'])
            vectors.append(
                {
                    'id': f"thought-{self.thought_id_timestamp}-{record['chunk']}",
                    'values': vector,
                    'metadata': {'text': record['text'], **record}
                }
            )

        upsert_response = memory.upsert(
            vectors=vectors,
            namespace=namespace,
        )
        return upsert_response


    # @tool("_update_memories", return_direct=True, args_schema=VectorDBInput)


    def _fetch_memories(self, observation: str, namespace:str) -> List[Document]:
        """Fetch related characteristics, preferences or dislikes for a user."""
        query_embedding = self.get_ada_embedding(observation)
        memory = self.init_pinecone(index_name=self.index)
        memory.query(query_embedding, top_k=1, include_metadata=True, namespace=namespace,
                          filter={'user_id': {'$eq': self.user_id}})
    #     return self.memory_retriever.get_relevant_documents(observation)
    def _compute_agent_summary(self, model_speed:str):
        """Computes summary for a person"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_characteristics}"
            + "{relevant_preferences}"
            + "{relevant_dislikes}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        print("Computing Agent Summary")
        self.init_pinecone(index_name=self.index)
        # The agent seeks to think about their core characteristics.
        relevant_characteristics = self._fetch_memories(f"Users core characteristics", namespace="PREFERENCE")
        relevant_preferences = self._fetch_memories(f"Users core preferences", namespace="PREFERENCE")
        relevant_dislikes = self._fetch_memories(f"Users core dislikes", namespace="PREFERENCE")
        if model_speed =='fast':
            output = self.replicate_llm(prompt)
            return output

        else:
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
            return chain.run(name= self.user_id, relevant_characteristics=relevant_characteristics, relevant_preferences=relevant_preferences, relevant_dislikes=relevant_dislikes).strip()



    def update_agent_preferences(self, preferences:str):
        """Serves to update agents preferences so that they can be used in summary"""

        prompt = """ The {name} has following {past_preference} and the new {preferences}
                Update user preferences and return a list of preferences
            Do not embellish.
            Summary: """
        self.init_pinecone(index_name=self.index)
        past_preference = self._fetch_memories(f"Users core preferences", namespace="PREFERENCE")
        prompt = PromptTemplate(input_variables=["name", "past_preference", "preferences"], template=prompt)
        prompt = prompt.format(name=self.user_id, past_preference= past_preference, preferences=preferences)
        return self._update_memories(prompt, namespace="PREFERENCE")

    def update_agent_taboos(self, dislikes:str):
        """Serves to update agents taboos so that they can be used in summary"""
        prompt =""" The {name} has following {past_dislikes} and the new {dislikes}
                Update user taboos and return a list of taboos
            Do not embellish.
            Summary: """
        self.init_pinecone(index_name=self.index)
        past_dislikes = self._fetch_memories(f"Users core dislikes", namespace="PREFERENCE")
        prompt = PromptTemplate(input_variables=["name", "past_dislikes", "dislikes"], template=prompt)
        prompt = prompt.format(name=self.user_id, past_dislikes= past_dislikes, dislikes=dislikes)
        return self._update_memories(prompt, namespace="PREFERENCE")


    def update_agent_traits(self, traits:str):
        """Serves to update agent traits so that they can be used in summary"""
        prompt =""" The {name} has following {past_traits} and the new {traits}
                Update user traits and return a list of traits
            Do not embellish.
            Summary: """
        self.init_pinecone(index_name=self.index)
        past_traits = self._fetch_memories(f"Users core dislikes", namespace="PREFERENCE")
        prompt = PromptTemplate(input_variables=["name", "past_traits", "traits"], template=prompt)
        prompt = prompt.format(name=self.user_id, past_traits= past_traits, traits=traits)
        return self._update_memories(prompt, namespace="PREFERENCE")


    def update_agent_summary(self):
        """Serves to update agent traits so that they can be used in summary"""
        summary = self._compute_agent_summary()
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
    def recipe_generation(self, factors:dict, model_speed:str):
        """Generates a recipe solution in json"""
        import time

        start_time = time.time()
        prompt = """
                Help me choose what recipe to eat or make for my next meal.     
                There are {% for factor, value in factors.items() %}'{{ factor }}'{% if not loop.last %}, {% endif %}{% endfor %} factors I want to consider.
                {% for factor, value in factors.items() %}
                For '{{ factor }}', I want the meal to be '{{ value }}' points on a scale of 1 to 100 points{% if not loop.last %}.{% else %}.{% endif %}
                {% endfor %}
                Instructions and ingredients should be detailed.
                 Answer a condensed JSON with no whitespaces that contains the following keys and values for every recipe in the list of field "recipes":
                 "title", "rating", "prep_time", "cook_time", "description", "ingredients", "instructions".  After the JSON output, don't explain or write anything
        """
        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(factors=factors)
        complete_query = str(agent_summary) + output
        complete_query = PromptTemplate.from_template(complete_query)

        if model_speed =='fast':
            output = self.replicate_llm(output)
            return output
        else:
            chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
            chain_result = chain.run(prompt=complete_query, name=self.user_id).strip()
            end_time = time.time()
            vectorstore: Pinecone = Pinecone.from_existing_index(
                index_name=self.index,
                embedding=OpenAIEmbeddings(),
                # filter={'user_id': {'$eq': self.user_id}},
                namespace='RESULT'
            )
            from datetime import datetime
            retriever = vectorstore.as_retriever()
            retriever.add_documents([Document(page_content=chain_result,
                                              metadata={'inserted_at': datetime.now(), "text": chain_result,
                                                        'user_id': self.user_id}, namespace="RESULT")])

            execution_time = end_time - start_time
            print("Execution time: ", execution_time, " seconds")
            return chain_result
    @post_execution(_compute_agent_summary)
    def goal_generation(self, factors: dict, model_speed:str):
        """Serves to optimize agent goals"""

        prompt = """
              Based on all the history and information of this user, suggest mind map that would have four decision points that are personal to him that he should apply to optimize his decision choices related to food. It must be food and nutrition related. 
              The cuisine should be one of the points, and goal should contain one or maximum three words. If user preferences don't override it, and if it includes time elemeent, it should be called "Time to make", if it descibes complexity of the dish, it should be  "Complexity", and if descibing food content it should be called "Macros"
              Answer a condensed JSON with no whitespaces. The structure should only contain a list of goals under field "goals". After the JSON output, don't explain or write anything.
            """

        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(factors=factors)
        print("HERE IS THE AGENT SUMMARY", agent_summary)
        print("HERE IS THE TEMPLATE", output)
        complete_query = str(agent_summary) + output
        complete_query = PromptTemplate.from_template(complete_query)
        if model_speed =='fast':
            output = self.replicate_llm(output)
            return output
        else:
            chain = LLMChain(llm=self.llm,  prompt=complete_query, verbose=self.verbose)
            chain_result = chain.run(prompt=complete_query).strip()
            print("RESULT IS ", chain_result)
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
            return chain_result

    def sub_goal_generation(self, factors: dict, model_speed:str):
        """Serves to generate sub goals for the user and drill down into it"""

        prompt = """
            Base
            d on all the history and information of this user, GOALS PROVIDED HERE  {% for factor in factors %} '{{ factor['name'] }}'{% if not loop.last %}, {% endif %}{% endfor %} 
             provide a mind map representation of the secondary nodes that can be used to narrow down the choice better.It needs to be food and nutrition related. Each of the results should have 4 sub nodes.
            Answer a condensed JSON with no whitespaces. The strucuture should only contain the list of subgoal items under field "sub_goals".
            Every subgoal should have a "goal_name" refers to the goal and the list of subgoals with "name" and a "amount" should be shown as a range from 0 to 100, with a value chosen explicilty and shown based on the personal preferences of the user.  
            After the JSON output, don't explain or write anything
            """

        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(factors=factors)
        print("HERE IS THE AGENT SUMMARY", agent_summary)
        print("HERE IS THE TEMPLATE", output)
        complete_query = str(agent_summary) + output
        complete_query = PromptTemplate.from_template(complete_query)
        if model_speed =='fast':
            output = self.replicate_llm(output)
            return output
        else:
            chain = LLMChain(llm=self.llm,  prompt=complete_query, verbose=self.verbose)
            chain_result = chain.run( prompt=complete_query).strip()
            vectorstore: Pinecone = Pinecone.from_existing_index(
                index_name=self.index,
                embedding=OpenAIEmbeddings(),
                # filter={'user_id': {'$eq': self.user_id}},
                namespace='SUBGOAL'
            )
            from datetime import datetime
            retriever = vectorstore.as_retriever()
            retriever.add_documents([Document(page_content=chain_result,
                                              metadata={'inserted_at': datetime.now(), "text": chain_result,
                                                        'user_id': self.user_id}, namespace="SUBGOAL")])
            print("RESULT IS ", chain_result)
            return chain_result

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
    def restaurant_generation(self, factors: dict, model_speed:str):
        """Serves to suggest a restaurant to the agent"""

        prompt = """
              Based on the following factors, There are {% for factor, value in factors.items() %}'{{ factor }}'{% if not loop.last %}, {% endif %}{% endfor %} factors I want to consider.
                {% for factor, value in factors.items() %}
                For '{{ factor }}', I want the meal to be '{{ value }}' points on a scale of 1 to 100 points{% if not loop.last %}.{% else %}.{% endif %}
                {% endfor %}
                Determine the type of restaurant you should offer to a customer. Make the recomendation very short and to a point, as if it is something you would type on google maps
            """

        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(factors=factors)
        complete_query = str(agent_summary) + output
        # print('HERE IS THE COMPLETE QUERY', complete_query)
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
    async def delivery_generation(self, factors: dict, zipcode:str, model_speed:str):
        """Serves to optimize agent delivery recommendations"""

        prompt = """
              Based on the following factors, There are {% for factor, value in factors.items() %}'{{ factor }}'{% if not loop.last %}, {% endif %}{% endfor %} factors I want to consider.
                {% for factor, value in factors.items() %}
                For '{{ factor }}', I want the meal to be '{{ value }}' points on a scale of 1 to 100 points{% if not loop.last %}.{% else %}.{% endif %}
                {% endfor %}
                Determine the type of food you would want to recommend to the user, that is commonly ordered online. It should of type of food offered on a delivery app similar to burger or pizza, but it doesn't have to be that. 
                The response should be very short
            """

        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(factors=factors)
        complete_query = str(agent_summary) + output
        complete_query = PromptTemplate.from_template(complete_query)
        chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
        chain_result = chain.run(prompt=complete_query).strip()
        print("HERE IS THE PROMPT", chain_result)
        import asyncio
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
                namespace_val = "SUBGOAL"

            print("HERE IS THE NAMESPACE", namespace_val)
            vectorstore: Pinecone = Pinecone.from_existing_index(
                index_name=self.index,
                embedding=OpenAIEmbeddings(),
                # filter={'user_id': {'$eq': self.user_id}},
                namespace=namespace_val
            )
            from datetime import datetime
            retriever = vectorstore.as_retriever()
            retriever.search_kwargs = {'filter': {'user_id': {'$eq': self.user_id}}} # filter by user_id
            print(retriever.get_relevant_documents(summary_action))
            answer_response = retriever.get_relevant_documents(summary_action)
            answer_response.sort(key=lambda doc: doc.metadata.get('inserted_at') if 'inserted_at' in doc.metadata else datetime.min, reverse=True)
            # The most recent document is now the first element of the list.
            try:
                most_recent_document = answer_response[0]
            except IndexError:
                return {"error": "No document found for this user. Make sure that a query is appropriate"}
            escaped_content = most_recent_document.page_content.replace("{", "{{").replace("}", "}}")
            optimization_prompt = """Based on the query: {query} change and update appropriate of the following original: {{results}}
            Answer a condensed JSON with no whitespaces. The strucuture should only contain the list of subgoal items under field "sub_goals".
            Every subgoal should have a "goal_name" refers to the goal and the list of subgoals with "name" and a "amount" should be shown as a range from 0 to 100, with a value chosen explicilty and shown based on the personal preferences of the user.  
            After the JSON output, don't explain or write anything:"""

            optimization_prompt = Template(optimization_prompt)
            optimization_output = optimization_prompt.render( results=escaped_content)
            prompt_template = PromptTemplate(input_variables=["query"], template=optimization_output)
            review_chain = LLMChain(llm=self.llm, prompt=prompt_template)
            output = review_chain.run(query=summary_action).strip()
            print("HERE IS THE OUTPUT", output)
            return output
            # overall_chain = SimpleSequentialChain(chains=[chain, review_chain], verbose=True)
            # final_output = overall_chain.run('bu').strip()
            # print("HERE IS THE FINAL OUTPUT", final_output)
            # json_data = json.dumps(final_output)
            # return json_data







            # chain_result = chain.run(prompt=complete_query).strip()
            # json_data = json.dumps(chain_result)
            # return json_data
            # self._update_memories(observation="Man walks in a forest", namespace="test_namespace")
            template = """
            {summaries}
            {question}
            """

            # embeddings = OpenAIEmbeddings()
            # embeddings.embed_documents(["man walks in the forest", "horse walks in the field"])
            from langchain.chains import RetrievalQAWithSourcesChain
            from langchain.chains import RetrievalQA
            # llm = OpenAIEmbeddings()

            # from langchain.retrievers import TimeWeightedVectorStoreRetriever
            # retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.0000000000000000000000001,
            #                                              k=1)
            # yesterday = datetime.now() - timedelta(days=1)
            # #retriever.add_documents([Document(page_content="hello majmune", metadata={"last_accessed_at": yesterday,"text": "hello majmune",'user_id': self.user_id}, namespace="test_namespace")])
            # # retriever.add_documents([Document(page_content="hello foo", metadata={ "text": "hello foo"} ,namespace="test_namespace")])
            # print(retriever.get_relevant_documents("hello majmune"))
            from langchain.chains.qa_with_sources import load_qa_with_sources_chain
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


if __name__ == "__main__":
    agent = Agent()
    # agent.goal_optimization(factors={}, model_speed="slow")
    # agent._update_memories("lazy, stupid and hungry", "TRAITS")
    agent.voice_input("I need your help, I need to add weight loss as a goal ", model_speed="slow")
    # agent.goal_generation( {    'health': 85,
    # 'time': 75,
    # 'cost': 50}, model_speed="slow")