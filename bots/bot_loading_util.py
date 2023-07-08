from bot_extension import AppAgent
import  yaml
import sys
sys.path.append('../')
with open('../assistant_templates.yaml', 'r') as file:
    data = yaml.safe_load(file)

def _load_extension( object_type:str, object_value:str):
    naval_chat_bot = AppAgent()
    naval_chat_bot.add(object_type, object_value)

# Directly access the 'chatbot' role
chatbot_categories = data['chatbot']['categories']

# Iterate through the categories and resources
for category in chatbot_categories:
    for resource in category['resources']:
        resource_type = resource['type']
        resource_url = resource['url']
        _load_extension(resource_type, resource_url)

