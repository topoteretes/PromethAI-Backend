from langchain.document_loaders import PyPDFLoader


import pinecone
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from langchain.agents import initialize_agent
from langchain.tools import  tool
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from pydantic import BaseModel, Field,  parse_obj_as
import re
from jinja2 import Template
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.schema import Document
from langchain.chains import SimpleSequentialChain
from langchain.chains.openai_functions import (
    create_openai_fn_chain, create_structured_output_chain
)
from langchain.schema import HumanMessage, SystemMessage
import os
import fastjsonschema

import json
from langchain.tools import GooglePlacesTool
import tiktoken
import asyncio
import logging
from langchain.chat_models import ChatOpenAI, ChatLiteLLM
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents import AgentType
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from typing import Optional

# redis imports for cache
import langchain
from langchain.callbacks import get_openai_callback

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
from langchain.llms import Replicate
from redis import Redis
from langchain.cache import RedisCache
import os



logging.basicConfig(level=logging.INFO)
# from langchain import llm_cache


# langchain.llm_cache = RedisCache(redis_=Redis(host="redis", port=6379, db=0))
# logging.info("Using redis cache")


if os.getenv("AWS_ENV", "") == "dev":
    REDIS_HOST = os.getenv(
        "REDIS_HOST",
        "promethai-dev-backend-redis-repl-gr.60qtmk.ng.0001.euw1.cache.amazonaws.com",
    )
    langchain.llm_cache = RedisCache(
        redis_=Redis(host="promethai-dev-backend-redis-repl-gr.60qtmk.ng.0001.euw1.cache.amazonaws.com", port=6379,
                     db=0))
    logging.info("Using redis cache for DEV")
elif os.getenv("AWS_ENV", "") == "prd":
    REDIS_HOST = os.getenv(
        "REDIS_HOST",
        "promethai-prd-backend-redis-repl-gr.60qtmk.ng.0001.euw1.cache.amazonaws.com",
    )
    langchain.llm_cache = RedisCache(
        redis_=Redis(host="promethai-prd-backend-redis-repl-gr.60qtmk.ng.0001.euw1.cache.amazonaws.com", port=6379,
                     db=0))
    logging.info("Using redis cache for PRD")
else:
    pass


class Agent:
    load_dotenv()
    OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4"
    GPLACES_API_KEY = os.getenv("GPLACES_API_KEY", "")
    ZAPIER_NLA_API_KEY = os.environ["ZAPIER_NLA_API_KEY"] = os.environ.get(
        "ZAPIER_NLA_API_KEY", ""
    )
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "")
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")

    def __init__(
            self,
            table_name=None,
            user_id: Optional[str] = "676",
            session_id: Optional[str] = None,
    ) -> None:
        self.table_name = table_name
        self.user_id = user_id
        self.session_id = session_id
        # self.memory = None
        self.thought_id_timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[
                                    :-3
                                    ]  # Timestamp with millisecond precision
        self.last_message = ""
        self.openai_model35 = "gpt-3.5-turbo-16k-0613"
        self.openai_model4 = "gpt-4-0613"
        self.llm = ChatLiteLLM(
            temperature=0.0,
            max_tokens=1500,
            openai_api_key=self.OPENAI_API_KEY,
            model_name=self.openai_model35,
            cache=False,
        )
        self.llm35_fast = ChatLiteLLM(
            temperature=0.2,
            max_tokens=550,
            openai_api_key=self.OPENAI_API_KEY,
            model_name=self.openai_model35,
            cache=False,
        )
        self.llm_fast = ChatLiteLLM(
            temperature=0.0,
            max_tokens=700,
            openai_api_key=self.OPENAI_API_KEY,
            model_name=self.openai_model4,
            cache=False,
        )
        self.llm35 = ChatLiteLLM(
            temperature=0.0,
            max_tokens=1200,
            openai_api_key=self.OPENAI_API_KEY,
            model_name=self.openai_model35,
            cache=False,
        )
        # self.llm = ChatOpenAI(temperature=0.0,max_tokens = 1500, openai_api_key = self.OPENAI_API_KEY, model_name="gpt-4")
        self.replicate_llm = ChatLiteLLM(
            model="replicate/vicuna-13b:a68b84083b703ab3d5fbf31b6e25f16be2988e4c3e21fe79c2ff1c18b99e61c1",
            api_key=self.REPLICATE_API_TOKEN,
        )
        self.verbose: bool = True
        self.openai_temperature = 0.0
        self.index = "my-agent"

    def clear_cache(self):
        langchain.llm_cache.clear()

    def set_user_session(self, user_id: str, session_id: str) -> None:
        self.user_id = user_id
        self.session_id = session_id

    def get_ada_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(
            input=[text], model="text-embedding-ada-002", api_key=OPENAI_API_KEY
        )["data"][0]["embedding"]

    def init_pinecone(self, index_name):
        load_dotenv()
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
        PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "")
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
        return pinecone.Index(index_name)

    def _simple_test(self):
        # langchain.llm_cache = RedisCache(redis_=Redis(host='0.0.0.0', port=6379, db=0))
        with get_openai_callback() as cb:
            # langchain.llm_cache = RedisCache(redis_=Redis(host='0.0.0.0', port=6379, db=0))
            prompt = """ How long does it take to go to the moon on foot """
            prompt = PromptTemplate.from_template(prompt)
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
            chain_result = chain.run(prompt=prompt, name=self.user_id).strip()
            print(cb)

            return chain_result

    # create the length function
    def tiktoken_len(self, text):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)

    # class VectorDBInput(BaseModel):
    #     observation: str = Field(description="should be what we are inserting into the memory")
    #     namespace: str = Field(description="should be the namespace of the VectorDB")
    # @tool("_update_memories", return_direct=True, args_schema = VectorDBInput)

    # def insert_documents(self, documents, namespace):
    #     from datetime import datetime
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
    #                 },
    #                 namespace=namespace,
    #             )
    #         ]
    #     )
    def _update_memories(self, observation: str, namespace: str, page: str = "", source: str = "") -> None:
        """Update related characteristics, preferences or dislikes for a user."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.init_pinecone(index_name=self.index)
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

    class FetchMemories(BaseModel):
        observation: str = Field(
            description="observation we want to fetch from vectordb"
        )

    def _fetch_memories(self, observation: str, namespace: str) -> dict[str, str] | str:
        """Fetch related characteristics, preferences or dislikes for a user."""

        self.init_pinecone(index_name=self.index)
        vectorstore: Pinecone = Pinecone.from_existing_index(
            index_name=self.index, embedding=OpenAIEmbeddings(), namespace=namespace
        )
        retriever = vectorstore.as_retriever()
        retriever.search_kwargs = {"filter": {"user_id": {"$eq": self.user_id}}}
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

    def _compute_agent_summary(self, model_speed: str):
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

        relevant_preferences = self._fetch_memories(
            f"Users core preferences", namespace="PREFERENCES"
        )
        relevant_dislikes = self._fetch_memories(
            f"Users core dislikes", namespace="PREFERENCES"
        )
        print(relevant_dislikes)
        print(relevant_preferences)

        if model_speed == "fast":
            output = self.replicate_llm(prompt)
            return output

        else:
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
            chain_results = chain.run(
                name=self.user_id,
                relevant_preferences=relevant_preferences,
                relevant_dislikes=relevant_dislikes,
            ).strip()
            print(chain_results)
            return chain_results

    def update_agent_preferences(self, preferences: str):
        """Serves to update agents preferences so that they can be used in summary"""

        prompt = """ The {name} has following {past_preference} and the new {preferences}
                Update user preferences and return a list of preferences
            Do not embellish.
            Summary: """
        self.init_pinecone(index_name=self.index)
        past_preference = self._fetch_memories(
            f"Users core preferences", namespace="PREFERENCE"
        )
        prompt = PromptTemplate(
            input_variables=["name", "past_preference", "preferences"], template=prompt
        )

        # prompt = prompt.format(name=self.user_id, past_preference= past_preference, preferences=preferences)
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        chain_result = chain.run(
            prompt=prompt,
            past_preference=past_preference,
            preferences=preferences,
            name=self.user_id,
        ).strip()
        print(chain_result)
        return self._update_memories(chain_result, namespace="PREFERENCES")

    def update_agent_taboos(self, dislikes: str):
        """Serves to update agents taboos so that they can be used in summary"""
        prompt = """ The {name} has following {past_dislikes} and the new {dislikes}
                Update user taboos and return a list of dislikes
            Do not embellish.
            Summary: """
        self.init_pinecone(index_name=self.index)
        past_dislikes = self._fetch_memories(
            f"Users core dislikes", namespace="PREFERENCES"
        )
        prompt = PromptTemplate(
            input_variables=["name", "past_dislikes", "dislikes"], template=prompt
        )
        # prompt = prompt.format(name=self.user_id, past_dislikes= past_dislikes, dislikes=dislikes)
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        chain_result = chain.run(
            prompt=prompt,
            name=self.user_id,
            past_dislikes=past_dislikes,
            dislikes=dislikes,
        ).strip()
        return self._update_memories(chain_result, namespace="PREFERENCES")

    def update_agent_traits(self, traits: str):
        """Serves to update agent traits so that they can be used in summary"""
        prompt = """ The {name} has following {past_traits} and the new {traits}
                Update user traits and return a list of traits
            Do not embellish.
            Summary: """
        self.init_pinecone(index_name=self.index)

        past_traits = self._fetch_memories(
            f"Users core dislikes", namespace="PREFERENCES"
        )
        prompt = PromptTemplate(
            input_variables=["name", "past_traits", "traits"], template=prompt
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        chain_result = chain.run(
            prompt=prompt, past_traits=past_traits, traits=traits, name=self.user_id
        ).strip()
        return self._update_memories(chain_result, namespace="PREFERENCES")

    def update_agent_summary(self, model_speed):
        """Serves to update agent traits so that they can be used in summary"""
        summary = self._compute_agent_summary(model_speed=model_speed)
        return self._update_memories(summary, namespace="SUMMARY")

    def prompt_correction(self, prompt_source: str, model_speed: str):
        """Makes the prompt gramatically correct"""

        prompt = """ Gramatically and logically correct sentence: {{prompt_source}} . Return only the corrected sentence, no abbreviations, using same words if it is logical. Do not mention explicitly rules given in prompt. """
        template = Template(prompt)
        output = template.render(prompt_source=prompt_source)
        complete_query = PromptTemplate.from_template(output)

        chain = LLMChain(
            llm=self.llm, prompt=complete_query, verbose=self.verbose
        )
        chain_result = chain.run(prompt=complete_query, name=self.user_id).strip()
        json_data = json.dumps(chain_result)
        return json_data

    async def solution_generation(self, prompt: str, prompt_template: str = None, json_example: str = None,
                                  model_speed: str = None):
        """Generates a recipe solution in json"""

        if prompt_template is None:
            prompt_base = """ Create a food recipe based on the following prompt: '{{prompt}}'. Instructions and ingredients should have medium detail.
                Answer a condensed valid JSON in this format: {{ json_example}}  Do not explain or write anything else."""
        else:
            prompt_base = prompt_template

        if json_example is None:
            json_example = """{"recipes":[{"title":"value","rating":"value","prep_time":"value","cook_time":"value","description":"value","ingredients":["value"],"instructions":["value"]}]}"""
        else:
            json_example = json_example

        # json_example = str(json_example).replace("{", "{{").replace("}", "}}")
        # template = Template(prompt_base)
        # output = template.render(prompt=prompt
        #                          , json_example=json_example)
        # complete_query = output
        # complete_query = PromptTemplate.from_template(complete_query)

        # Define the response schema
        class Recipe(BaseModel):
            """Schema for an individual recipe."""
            title: str = Field(..., description="Title of the recipe")
            rating: str = Field(None, description="Recipe rating")
            prep_time: str = Field(None, description="Time to prepare recipe")
            cook_time: str = Field(None, description="Time to cook recipe")
            description: str = Field(None, description="Description of recipe")
            ingredients: List[str] = Field(None, description="All recipe ingredients")
            instructions: List[str] = Field(None, description="All recipe instructions for making a recipe")

        class RecordRecipe(BaseModel):
            """Schema for the record containing a list of recipes."""
            recipes: List[Recipe] = Field(..., description="List of recipes")

        prompt_msgs = [
            SystemMessage(
                content="You are a world class algorithm for creating recipes"
            ),
            HumanMessage(content="Create a food recipe based on the following prompt:"),
            HumanMessagePromptTemplate.from_template("{input}"),
            HumanMessage(content="Tips: Make sure to answer in the correct format"),
        ]
        prompt_ = ChatPromptTemplate(messages=prompt_msgs)
        chain = create_structured_output_chain(RecordRecipe, self.llm35, prompt_, verbose=True)
        from langchain.callbacks import get_openai_callback
        with get_openai_callback() as cb:
            output = await chain.arun(input=prompt)
            print(cb)
        # output = json.dumps(output)
        my_object = parse_obj_as(RecordRecipe, output)
        return my_object.dict()



    async def solution_name_generation(self, prompt: str, prompt_template: str = None, json_example: str = None,
                                       model_speed: str = None):
        """Generates a single recipe solution and returns the recipe title as a string."""

        prompt_ = """Create a food recipe based on the following prompt: {{prompt}} Return just a concise recipe title. Do not explain or write anything else."""

        template = Template(prompt_)
        output = template.render(prompt=prompt)
        complete_query = PromptTemplate.from_template(output)

        chain = LLMChain(
            llm=self.llm35_fast, prompt=complete_query, verbose=self.verbose
        )
        chain_result = await chain.arun(prompt=complete_query, name=self.user_id)
        logging.info("Here is the chain result ",chain_result)
        # json_data = json.dumps(chain_result)
        return str(chain_result)

        # if model_speed == "fast":
        #     output = self.replicate_llm(output)
        #     return output
        # else:
        #     chain = LLMChain(
        #         llm=self.llm, prompt=complete_query, verbose=self.verbose
        #     )
        #     chain_result = chain.run(prompt=complete_query, name=self.user_id).strip()
        #     #
        #     # vectorstore: Pinecone = Pinecone.from_existing_index(
        #     #     index_name=self.index,
        #     #     embedding=OpenAIEmbeddings(),
        #     #     namespace='RESULT'
        #     # )
        #     # from datetime import datetime
        #     # retriever = vectorstore.as_retriever()
        #     # retriever.add_documents([Document(page_content=chain_result,
        #     #                                   metadata={'inserted_at': datetime.now(), "text": chain_result,
        #     #                                             'user_id': self.user_id}, namespace="RESULT")])
        #     logging.info("HERE IS THE CHAIN RESULT", chain_result)
        #     return chain_result

    def extract_json(self, data):
        json_start = data.find("{")
        json_end = data.rfind("}") + 1
        json_data = data[json_start:json_end]
        try:
            return json.loads(json_data)  # if successful, return Python dict
        except json.JSONDecodeError:
            return None  # if unsuccessful, return None

    async def async_generate(
            self,
            prompt_template_base,
            base_category,
            base_value,
            list_of_items,
            assistant_category,
    ):
        """Generates an individual solution choice"""
        json_example = """ {"category":"time","options":[{"category":"quick","options":[{"category":"1 min"},{"category":"10 mins"},{"category":"30 mins"}]},{"category":"slow","options":[{"category":"60 mins"},{"category":"120 mins"},{"category":"180 mins"}]}]}"""
        #
        list_of_items = [
            item for item in list_of_items if item != [base_category, base_value]
        ]
        # logging.info("list of items", list_of_items)
        # try:
        #     list_as_string = str(list_of_items[0]).strip("[]")
        # except:
        #     list_as_string = str(list_of_items)
        # # agent_summary = agent_summary.split('.', 1)[0]
        json_example = json_example.replace("{", "{{").replace("}", "}}")
        # template = Template(prompt_template_base)
        # output = template.render(
        #     base_category=base_category,
        #     base_value=base_value,
        #     json_example=json_example,
        #     assistant_category=assistant_category,
        #     exclusion_categories=list_as_string,
        # )
        # complete_query = PromptTemplate.from_template(output)
        #
        # chain = LLMChain(llm=self.llm_fast, prompt=complete_query, verbose=self.verbose)
        # chain_result = await chain.arun(prompt=complete_query, name=self.user_id)
        # json_o = json.loads(chain_result)
        # value_list = [{"category": value} for value in base_value.split(",")]
        # # json_o["options"].append({"category": "Your preferences", "options": value_list})
        # chain_result = json.dumps(json_o)
        # print("FINAL CHAIN", chain_result)
        # return chain_result
        list_of_items = str(list_of_items)
        #
        #
        class FoodOption(BaseModel):
            category: str = Field(...,
                                  description="Specific food option category, e.g., 'Italian', 'Gluten-free', 'Outdoor seating'")

        class CategoryOption(BaseModel):
            category: str = Field(...,
                                  description="Main category, e.g., 'Cuisine', 'Dietary Restrictions', 'Atmosphere'")
            options: List[FoodOption] = Field([], description="List of possible options for this main category")

        class Response(BaseModel):
            category: str = Field(..., description="Main classification, e.g., 'Location', 'Price Range'")
            options: List[CategoryOption] = Field([], description="An array of category option objects.")
        #
        system_message = f"You are a world class algorithm for decomposing human " \
                         f"thoughts into decision trees on {assistant_category}. "
        guidance_query = f"The request:"
        prompt_msgs = [
            SystemMessage(
                content=system_message
            ),
            HumanMessage(content=guidance_query),
            HumanMessagePromptTemplate.from_template("{input}"),
            HumanMessage(content=f"Tips: Make sure to answer in the correct format"),
            HumanMessage(content=f"Tips: Must include the following as a category: {base_value} and exclude {list_of_items}"),
            HumanMessage(content=f" Tips: Look at this json as example: {json_example}"),
            HumanMessage(content=f"Tips: Escape possesive apostrophes with a backslash, e.g., 'John\\'s' "),
            # HumanMessage(content=f"Tips: Exclude the following categories: {list_of_items}"),
        ]
        prompt_ = ChatPromptTemplate(messages=prompt_msgs)
        chain = create_structured_output_chain(Response, self.llm35, prompt_, verbose=True)
        output = await chain.arun(input=f"""Decompose decision point '{ base_category }' into three categories with the same or lower granularity and must include '{base_value}'.
        Provide three sub-categories that specify the decision point better.""" )
        # from pydantic import BaseModel, parse_raw
        # Convert the dictionary to a Pydantic object
        my_object = parse_obj_as(Response, output)
        data = my_object.dict()
        return str(data).replace("'", '"')

    async def generate_concurrently(self, base_prompt, assistant_category,load_defaults=True):
        """Generates an async solution group"""


        list_of_items = [item.split("=") for item in base_prompt.split(";")]
        prompt_template_base = """ Decompose decision point '{{ base_category }}' into three categories the same level as value '{{base_value}}'  definitely including '{{base_value}} ' but not including  {{exclusion_categories}}. Make sure choices further specify the  '{{ base_category }}' category  where AI is helping person in choosing {{ assistant_category }}.
        Provide three sub-options that further specify the particular category better. Generate very short json, do not write anything besides json, follow this json property structure : {{json_example}}"""
        list_of_items = base_prompt.split(";")

        # If there is no ';', split on '=' instead
        if len(list_of_items) == 1:
            list_of_items = [list_of_items[0].split("=")]
        else:
            list_of_items = [item.split("=") for item in list_of_items]
            # Remove  value
            print("LIST OF ITEMS", list_of_items)
            logging.info("LIST OF ITEMS", str(list_of_items))
        tasks = [
            self.async_generate(
                prompt_template_base,
                base_category,
                base_value,
                list_of_items,
                assistant_category,
            )
            for base_category, base_value in list_of_items
        ]
        results = await asyncio.gather(*tasks)

        def replace_underscores(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "category" and isinstance(value, str):
                        data[key] = value.replace("_", " ")
                    else:
                        replace_underscores(value)
            elif isinstance(data, list):
                for item in data:
                    replace_underscores(item)




        if len(results) == 1:
            logging.info("HERE ARE THE valid RESULTS %s", str(results))
            results_list = [json.loads(results[0])]

        else:
            logging.info("HERE ARE THE valid RESULTS %s", len(results))
            print("HERE ARE THE valid RESULTS %s", len(results))
            # Parse each JSON string and add it to a list
            results = [
                result[result.find("{"): result.rfind("}") + 1] for result in results
            ]
            results_list = [json.loads(result) for result in results]

        replace_underscores(results_list)
        combined_json = {"results": results_list}

        def load_schema(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)


        try:
            schema_path = os.path.join(os.path.dirname(__file__), '..', 'validations', 'schema',
                                       'decompose_categories.json')
            primary_schema = load_schema(schema_path)
            validate = fastjsonschema.compile(primary_schema)
            logging.info("HERE SOME RESULTS %s", str({"response":combined_json}))
            validate({"response":combined_json})
            return combined_json
        except fastjsonschema.exceptions.JsonSchemaException as e:
            logging.info("HERE ARE THE  ERRORS %s", str(e))
            schema_path = os.path.join(os.path.dirname(__file__), '..', 'validations', 'defaults',
                                       'categories_defaults.json')
            combined_json = load_schema(schema_path)
            return combined_json



    def _loader(self, path: str, namespace: str):

        loader = PyPDFLoader("../document_store/nutrition/Human_Nutrition.pdf")
        pages = loader.load_and_split()

        print("PAGES", pages[0])

        for page in pages:
            self._update_memories(page.page_content, namespace, page.metadata["page"], page.metadata["source"])
        return "Success"
        # print(type(pages))

    def _process_pref(self, data):
        for result in data["response"]["results"]:
            # Check if preference is empty and options exist
            if not result["preference"] and result["options"]:
                # Get the second nested category value
                second_category = result["options"][0]["category"]
                # Assign it to the preference
                result["preference"] = [second_category]

        # Assuming data is a dictionary with the structure you described
        def remove_second_subnested_category(categories):
            for category_data in categories:
                if "options" in category_data:
                    # If there are options in the current category, check if there is a second subnested category and remove it if it exists
                    try:
                        category_data["options"].pop(1)
                    except IndexError:
                        pass

                    # Recursively iterate over all subcategories, if any
                    remove_second_subnested_category(category_data["options"])

        remove_second_subnested_category(data["response"]["results"])
        print("UPDATED OUTPUT", data)
        return data

        # Assuming you have the JSON data in the "data" variable

    def prompt_to_choose_tree(self, prompt: str, model_speed: str, assistant_category: str, load_defaults: bool = True):
        """Serves to generate agent goals and subgoals based on a prompt"""


        def load_schema(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)

        if load_defaults:
            schema_path = os.path.join(os.path.dirname(__file__), '..', 'validations', 'defaults',
                                       'categories_input_defaults.json')
            combined_json = json.dumps(load_schema(schema_path))
            return combined_json
        else:
            json_example = """ <category1>=<decision1>;<category2>=<decision2>..."""
            prompt_template = """
               Decompose {{ prompt_str }} statement into decision tree that take into account user summary information and related to {{ assistant_category }}. There should be three categories and one decision for each.  
               Categories should be logical and user friendly. Do not include budget, meal type, intake, personality, user summary, personal preferences.
               Decision should be one user can make in regards to {{ assistant_category }}. Present answer in one line and in property structure : {{json_example}}"""
            bb =  """Do not include budget, meal type, intake, personality, user summary, personal preferences, or update time to categories.  """

                # self.init_pinecone(index_name=self.index)
                # try:
                #     agent_summary = self._fetch_memories(
                #         f"Users core summary", namespace="SUMMARY"
                #     )
                #     print("HERE IS THE AGENT SUMMARY", agent_summary)
                #     agent_summary = str(agent_summary)
                #
                #     if (
                #             str(agent_summary)
                #             == "{'error': 'No document found for this user. Make sure that a query is appropriate'}"
                #     ):
                #         agent_summary = "None."
                # except:
                #     agent_summary = "None."
                #
                # import time
                # start_time = time.time()

                # agent_summary = agent_summary.split(".", 1)[0]
            template = Template(prompt_template)
            output = template.render(
                prompt_str=prompt,
                json_example=json_example,
                # user_summary=agent_summary,
                assistant_category=assistant_category,
                # nutritional_context=test_output['answer']
            )
            complete_query = output
            print("HERE IS THE COMPLETE QUERY", complete_query)
            complete_query = PromptTemplate.from_template(complete_query)


            chain = LLMChain(llm=self.llm_fast, prompt=complete_query, verbose=False)
            chain_result = chain.run(prompt=complete_query, name=self.user_id).strip()

            import re

            def add_space_to_camel_case(s):
                # Check if the string contains any uppercase letters
                if any(c.isupper() for c in s[1:]):  # We exclude the first character from the check
                    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)

                # Convert each word to title case
                return ' '.join([word.capitalize() for word in s.split()])

            chain_result= add_space_to_camel_case(chain_result)
            class Option(BaseModel):
                category: str = Field(..., description=" Each should have a 'category' (a specific choice like 'Under $25' or 'Red')")
                options: Optional[List] = Field([], description="Empty list")
            class Result(BaseModel):
                category: str = Field(None, description=" Specify the main classification (e.g., Price Range, Color, Size) in the 'category' field.")
                options: List[Option] = Field(None, description="An array of option objects.")
                preference: Optional[List] = Field([], description="Value of the first category")
            class Response(BaseModel):
                results: List[Result] = Field(None, description="List of the results of the decision tree")

            class Main(BaseModel):
                response: Response = Field(None, description="Complete decision tree response")

            system_message = f"You are a world class algorithm applying raw output to a schema " \
                             # f" into decision trees on {assistant_category}. "
            # guidance_query = f"Decompose sentences into decision trees on {assistant_category}. " \
            #                  f"Decompose the following statement:"

            guidance_query = f"Apply output and change it to a schema"
            prompt_msgs = [
                SystemMessage(
                    content=system_message
                ),
                HumanMessage(content=guidance_query),
                HumanMessagePromptTemplate.from_template("{input}"),
                HumanMessage(content=f"Tips: Make sure to answer in the correct format"),
                HumanMessage(content=f"Tips: Make sure lowest level options are an empty list "),
                HumanMessage(content=f"Tips: Make sure results have multiple categories on the same level ")
            ]
            prompt_ = ChatPromptTemplate(messages=prompt_msgs)
            chain = create_structured_output_chain(Main, self.llm35, prompt_, verbose=True)
            output = chain.run(input=chain_result)
            # from pydantic import BaseModel, parse_raw
            # Convert the dictionary to a Pydantic object
            my_object = parse_obj_as(Main, output)
            data = my_object.dict()
            logging.info("HERE IS THE inter RESULT", str(data).replace("'", '"'))
            print("HERE IS THE DICT", data)
            data_pr = self._process_pref(data)
            logging.info("HERE IS THE FINAL RESULT", str(data_pr).replace("'", '"'))
            combined_json = data_pr
            # combined_json = str(data_pr).replace("'", '"')


            try:

                schema_path = os.path.join(os.path.dirname(__file__), '..', 'validations', 'schema',
                                           'decompose_categories_input.json')
                primary_schema = load_schema(schema_path)
                validate = fastjsonschema.compile(primary_schema)
                logging.info("HERE SOME RESULTS %s", str(combined_json))
                try:
                    validate(combined_json)
                except:
                    validate(json.loads(combined_json))
                return str(combined_json).replace("'", '"')
            except:
                # logging.info("HERE ARE THE  ERRORS %s", str(e))
                schema_path = os.path.join(os.path.dirname(__file__), '..', 'validations', 'defaults',
                                           'categories_input_defaults.json')
                combined_json = json.dumps(load_schema(schema_path))
                return combined_json

    # def prompt_to_choose_tree(self, prompt: str, model_speed: str, assistant_category: str):
    #     """Serves to generate agent goals and subgoals based on a prompt"""
    #
    #     self.init_pinecone(index_name=self.index)
    #     vectorstore: Pinecone = Pinecone.from_existing_index(
    #         index_name=self.index, embedding=OpenAIEmbeddings(), namespace="NUTRITION_RESOURCE"
    #     )
    #     retriever = vectorstore.as_retriever()
    #     #
    #     # template = """
    #     #    {summaries}
    #     #    {question}
    #     #    """
    #     #
    #     # chain = RetrievalQAWithSourcesChain.from_chain_type(
    #     #     llm=OpenAI(temperature=0),
    #     #     chain_type="stuff",
    #     #     retriever=retriever,
    #     #     chain_type_kwargs={
    #     #         "prompt": PromptTemplate(
    #     #             template=template,
    #     #             input_variables=["summaries", "question"],
    #     #         ),
    #     #     },
    #     # )
    #     # test_output = chain(
    #     #     "Retireve and summarize releavant information from the following document. Turn it into into decision tree that take into account user summary information and related to food. Present answer in one line summary")
    #     # print("TEST OUTPUT", test_output['answer'])
    #
    #     # prompt_template = """Retireve and summarize releavant information from the following document
    #     #
    #     #
    #     # {text}
    #     #
    #     #
    #     # Turn it into into decision tree that take into account user summary information and related to {{ assistant_category }}.
    #     # Do not include budget, personality, user summary, personal preferences, or update time to categories. Do not include information about publisher or details. """
    #     # prompt_template = Template(prompt_template)
    #     #
    #     # prompt_template = prompt_template.render(
    #     #     original_prompt=prompt,
    #     #     assistant_category=assistant_category)
    #     # PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    #     # chain_summary = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce", return_intermediate_steps=True,
    #     #                              map_prompt=PROMPT, combine_prompt=PROMPT)
    #     # test_output = chain_summary({"input_documents": pages[1:20]},   return_only_outputs=True)
    #     #
    #     # print("TEST OUTPUT", test_output)
    #
    #     json_example = """ <category1>=<decision1>;<category2>=<decision2>..."""
    #     prompt_template = """Known user summary: '{{ user_summary }} '.
    #        Decompose {{ prompt_str }} statement into decision tree that take into account user summary information and related to {{ assistant_category }}.
    #        Do not include budget, meal type, intake, personality, user summary, personal preferences, or update time to categories.  Use the information to correct any major mistakes: {{nutritional_context}}
    #        Decision should be one user can make. Present answer in one line and in property structure : {{json_example}}"""
    #
    #     self.init_pinecone(index_name=self.index)
    #     try:
    #         agent_summary = self._fetch_memories(
    #             f"Users core summary", namespace="SUMMARY"
    #         )
    #         print("HERE IS THE AGENT SUMMARY", agent_summary)
    #         agent_summary = str(agent_summary)
    #
    #         if (
    #                 str(agent_summary)
    #                 == "{'error': 'No document found for this user. Make sure that a query is appropriate'}"
    #         ):
    #             agent_summary = "None."
    #     except:
    #         agent_summary = "None."
    #
    #     import time
    #     start_time = time.time()
    #
    #     agent_summary = agent_summary.split(".", 1)[0]
    #     template = Template(prompt_template)
    #     output = template.render(
    #         prompt_str=prompt,
    #         json_example=json_example,
    #         user_summary=agent_summary,
    #         assistant_category=assistant_category,
    #         # nutritional_context=test_output['answer']
    #     )
    #     complete_query = output
    #     print("HERE IS THE COMPLETE QUERY", complete_query)
    #     complete_query = PromptTemplate.from_template(complete_query)
    #     if model_speed == "fast":
    #         output = self.replicate_llm(output)
    #         json_data = json.dumps(output)
    #         return json_data
    #     else:
    #         chain = LLMChain(llm=self.llm_fast, prompt=complete_query, verbose=False)
    #         chain_result = chain.run(prompt=complete_query, name=self.user_id).strip()
    #         vectorstore: Pinecone = Pinecone.from_existing_index(
    #             index_name=self.index,
    #             embedding=OpenAIEmbeddings(),
    #             namespace="GOAL",
    #         )
    #         from datetime import datetime
    #         retriever = vectorstore.as_retriever()
    #         logging.info(str(chain_result))
    #         print("HERE IS THE CHAIN RESULT", chain_result)
    #         retriever.add_documents(
    #             [
    #                 Document(
    #                     page_content=chain_result,
    #                     metadata={
    #                         "inserted_at": datetime.now(),
    #                         "text": chain_result,
    #                         "user_id": self.user_id,
    #                     },
    #                     namespace="GOAL",
    #                 )
    #             ]
    #         )
    #     return chain_result.replace("'", '"')

    async def prompt_decompose_to_tree_categories(
            self, prompt: str, assistant_category, model_speed: str, load_defaults: bool=True
    ):
        """Serves to generate agent goals and subgoals based on a prompt"""
        def load_schema(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)

        if load_defaults:
            schema_path = os.path.join(os.path.dirname(__file__), '..', 'validations', 'defaults',
                                       'categories_defaults.json')
            combined_json = load_schema(schema_path)
            return combined_json
        else:
            combined_json = await self.generate_concurrently(prompt, assistant_category, load_defaults=load_defaults)
            return combined_json
        # async for result in self.generate_concurrently(prompt):
        #     yield result

    def prompt_to_update_meal_tree(
            self, category: str, from_: str, to_: str, model_speed: str
    ):
        self.init_pinecone(index_name=self.index)
        vectorstore: Pinecone = Pinecone.from_existing_index(
            index_name=self.index, embedding=OpenAIEmbeddings(), namespace="GOAL"
        )

        retriever = vectorstore.as_retriever()
        retriever.search_kwargs = {
            "filter": {"user_id": {"$eq": self.user_id}}
        }  # filter by user_id
        answer_response = retriever.get_relevant_documents("prompt")
        answer_response.sort(
            key=lambda doc: doc.metadata.get("inserted_at")
            if "inserted_at" in doc.metadata
            else datetime.min,
            reverse=True,
        )

        # The most recent document is now the first element of the list.
        try:
            most_recent_document = answer_response[0]
        except IndexError:
            return {
                "error": "No document found for this user. Make sure that a query is appropriate"
            }
        doc = most_recent_document.page_content
        json_str = doc.replace("'", '"')
        document = json.loads(json_str)
        matching_items = [
            item for item in document["tree"] if item["category"] == category
        ]
        sub_tree = matching_items[0] if matching_items else None
        sub_tree = json.dumps(sub_tree)
        escaped_content = sub_tree.replace("{", "{{").replace("}", "}}")
        logging.info(escaped_content)

        optimization_prompt = """Change the category: {{category}} based on {{from_}} to {{to_}}  change and update appropriate of the following original inluding the preference: {{results}}
         """

        optimization_prompt = Template(optimization_prompt)
        optimization_output = optimization_prompt.render(
            category=category, from_=from_, to_=to_, results=escaped_content
        )
        complete_query = PromptTemplate.from_template(optimization_output)
        # prompt_template = PromptTemplate(input_variables=["query"], template=optimization_output)
        review_chain = LLMChain(llm=self.llm, prompt=complete_query)
        review_chain_result = review_chain.run(
            prompt=complete_query, name=self.user_id
        ).strip()
        return review_chain_result.replace("'", '"')

    def extract_info(self, s: str):
        lines = s.split("\n")
        name = lines[0]
        address = lines[1].replace("Address: ", "")
        phone = lines[2].replace("Phone: ", "")
        website = lines[3].replace("Website: ", "")
        return {
            "name": name,
            "address": address,
            "phone": phone,
            "website": website,
        }

    async def restaurant_generation(self, prompt: str, prompt_template: str, json_example: str, model_speed: str):
        """Serves to suggest a restaurant to the agent"""

        if prompt:
            prompt = prompt
        else:
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
        restaurants = re.split(r"\d+\.", output)[1:3]
        # Create a list of dictionaries for each restaurant
        restaurant_list = [self.extract_info(r) for r in restaurants]
        print("HERE IS THE OUTPUT", restaurant_list)
        return restaurant_list

    # async def run_wolt_tool(self, zipcode, chain_result):
    #     from food_scrapers import  wolt_tool
    #     return wolt_tool.main(zipcode, chain_result)
    async def delivery_generation(self, prompt: str, zipcode: str, model_speed: str):
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

        output = await wolt_tool.main(zipcode=zipcode, prompt=chain_result)
        return output

    def add_zapier_calendar_action(self, prompt_base, token, model_speed: str):
        """Serves to add a calendar action to the user's Google Calendar account"""

        # try:
        ZAPIER_NLA_OAUTH_ACCESS_TOKEN = token
        zapier = ZapierNLAWrapper(
            zapier_nla_oauth_access_token=ZAPIER_NLA_OAUTH_ACCESS_TOKEN
        )
        toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
        agent = initialize_agent(
            toolkit.get_tools(),
            self.llm_fast,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
        # except:
        #     zapier = ZapierNLAWrapper()
        #     toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
        #     agent = initialize_agent(toolkit.get_tools(), self.llm_fast, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #                              verbose=True)

        template = """ Formulate the following statement into a calendar request containing time, title, details of the meeting: {prompt} """
        prompt_template = PromptTemplate(input_variables=["prompt"], template=template)
        # complete_query = PromptTemplate.from_template(output)
        chain = LLMChain(llm=self.llm, prompt=prompt_template, verbose=self.verbose)
        overall_chain = SimpleSequentialChain(chains=[chain, agent], verbose=True)
        outcome = overall_chain.run(prompt_base)
        print("HERE IS THE OUTCOME", outcome)
        return outcome

    def voice_text_input(self, query: str, model_speed: str):
        """Serves to generate sub goals for the user and or update the user's preferences"""

        class GoalWrapper(BaseModel):
            observation: str = Field(
                description="observation we want to fetch from vectordb"
            )

        @tool("goal_update_wrapper", args_schema=GoalWrapper, return_direct=True)
        def goal_update_wrapper(observation, args_schema=GoalWrapper):
            """Fetches data from the VectorDB and returns it as a python dictionary."""
            query = self._fetch_memories(observation, "GOAL")
            loop = asyncio.get_event_loop()
            res = loop.run_until_complete(
                self.prompt_decompose_to_meal_tree_categories(query, "slow")
            )
            loop.close()
            return res

        class UpdatePreferences(BaseModel):
            observation: str = Field(
                description="observation we want to fetch from vectordb"
            )

        @tool("preferences_wrapper", args_schema=UpdatePreferences, return_direct=True)
        def preferences_wrapper(observation, args_schema=UpdatePreferences):
            """Updates user preferences in the VectorDB."""
            return self._update_memories(observation, "PREFERENCES")

        agent = initialize_agent(
            llm=self.llm_fast,
            tools=[goal_update_wrapper, preferences_wrapper],
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=self.verbose,
        )

        prompt = """

            Based on all the history and information of this user, classify the following query: {query} into one of the following categories:
            1. Goal update , 2. Preference change,  3. Result change 4. Subgoal update  If the query is not any of these, then classify it as 'Other'
            Return the classification and a very short summary of the query as a python dictionary. Update or replace or remove the original factors with the new factors if it is specified.
            with following python dictionary format 'Result_type': 'Goal', "Result_action": "Goal changed", "value": "Diet added", "summary": "The user is updating their goal to lose weight"
            Make sure to include the factors in the summary if they are provided
            """

        template = Template(prompt)
        output = template.render(query=query)
        complete_query = output
        complete_query = PromptTemplate(
            input_variables=["query"], template=complete_query
        )
        summary_chain = LLMChain(
            llm=self.llm, prompt=complete_query, verbose=self.verbose
        )
        from langchain.chains import SimpleSequentialChain

        overall_chain = SimpleSequentialChain(
            chains=[summary_chain, agent], verbose=True
        )
        output = overall_chain.run(query)
        return output

    def fetch_user_summary(self, model_speed: str):
        """Serves to retrieve agent summary"""
        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        return agent_summary

    def _retrieve_summary(self):
        """Serves to retrieve agent summary"""
        self.init_pinecone(index_name=self.index)
        result = self._fetch_memories("Users core prompt", "GOAL")
        print(result)
        return result


if __name__ == "__main__":
    agent = Agent()
    # agent.prompt_correction(prompt_source="I would like a quicko veggiea meals under 25 near me and", model_speed="slow")
    # agent.goal_optimization(factors={}, model_speed="slow")
    # agent._update_memories("lazy, stupid and hungry", "TRAITS")
    # agent.update_agent_traits("His personality is greedy")
    # agent.update_agent_preferences("Alergic to corn")
    # agent.add_zapier_calendar_action("I would like to schedule 1 hour meeting tomorrow at 12 about brocolli", 'bla', 'BLA')
    # agent.update_agent_summary(model_speed="slow")
    # agent.solution_generation(prompt="I would like a healthy chicken meal over 125$", model_speed="slow")
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(agent.prompt_decompose_to_meal_tree_categories("diet=vegan;availability=cheap", "food", model_speed="slow"))
    # loop.close()
    # import asyncio
    #
    #
    # async def main():
    #     out = await agent.prompt_to_choose_tree(prompt="I want would like a quick veggie meal Vietnamese cuisine",
    #                                             assistant_category="food", model_speed="slow")
    #     # Rest of your code here
    #
    #
    # # Run the async function
    # asyncio.run(main())

    # print(result)
    # agent._test()
    # agent.update_agent_summary(model_speed="slow")
    # agent.voice_text_input("Core prompt ", model_speed="slow")
