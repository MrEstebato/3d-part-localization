import google.generativeai as genai
import os
import subprocess
import argparse
from prompts import PROMPT1
from datetime import datetime
import time
import json

with open("config.json", "r") as f:
    config = json.load(f)

genai.configure(api_key=config["api_key"])

def create_chat():
    model = genai.GenerativeModel(
        model_name=config["model_name"],
        system_instruction=PROMPT1,
        generation_config={"temperature": config["temperature"]}
    )
    return model.start_chat()


def get_pure_python(gemini_output):
    """Extrae el código Python puro de la respuesta de Gemini."""
    if gemini_output.startswith("```python"):
        gemini_output = gemini_output[len("```python"):].strip()
    if gemini_output.endswith("```"):
        gemini_output = gemini_output[:-len("```")].strip()
    return gemini_output


with open(config["input_file"], "r", encoding="utf-8") as f:
    json_str = f.read()

prompt = f"Using this JSON file {json_str} create a python cadquery code that allows me to create the STP model using the JSON instructions"

chat = create_chat()

response = chat.send_message(prompt)

pythonCode = get_pure_python(response.text)

with open(config["output_file"], "w") as f:
    f.write(pythonCode)