import google.generativeai as genai
import os
import subprocess
import argparse
from prompts import PROMPT
from datetime import datetime
import time
import json

with open("config.json", "r") as f:
    config = json.load(f)

genai.configure(api_key=config["api_key"])

def create_chat():
    model = genai.GenerativeModel(
        model_name=config["model_name"],
        system_instruction=PROMPT,
        generation_config={"temperature": config["temperature"]}
    )
    return model.start_chat()

def get_prompt():
    """ Convierte el prompt a un prompt de Gemini  """
    prompt = input("Welcome to your CAD creation assistant, please enter what you would like to modify or create: \n")
    
    return prompt


chat = create_chat()

prompt = get_prompt()

response = chat.send_message(prompt)

response = response.text.replace("```json","").strip("```")

with open(config["input_file"], "w") as f:
    f.write(response)
