import sys
from typing import Optional

sys.path.append('../llm_chains')
# from embedchain import EmbedChain

from llm_chains.chains import Agent
from embedchain import App

class AppAgent(App, Agent):
    def __init__(self, db=None, ef=None, table_name=None, user_id: Optional[str] = "676", session_id: Optional[str] = None):
        Agent.__init__(self, table_name, user_id, session_id)
        App.__init__(self, db)

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

