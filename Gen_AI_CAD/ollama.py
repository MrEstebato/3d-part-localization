import json
import requests
from prompts import PROMPT
from langchain_community.llms import Ollama

llm = Ollama(model="joshuaokolo/C3Dv0")

# Invoke the model with a query

prompt = input("Welcome to your CAD creation assistant, please enter what you would like to modify or create: \n")
response = llm.invoke(prompt)
print(response)

prompt = input(response + "\n")
response = llm.invoke(prompt)
print(response)