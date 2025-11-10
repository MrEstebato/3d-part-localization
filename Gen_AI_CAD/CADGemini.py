import google.generativeai as genai
import os
import subprocess
import argparse
from prompts import PROMPT1
from datetime import datetime
import time


genai.configure(api_key="llave")

def create_chat():
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=PROMPT1,
        generation_config={"temperature": 0}
    )
    return model.start_chat()


def get_pure_python(gemini_output):
    """Extrae el código Python puro de la respuesta de Gemini."""
    if gemini_output.startswith("```python"):
        gemini_output = gemini_output[len("```python"):].strip()
    if gemini_output.endswith("```"):
        gemini_output = gemini_output[:-len("```")].strip()
    return gemini_output


with open("modelcircle.json", "r", encoding="utf-8") as f:
    json_str = f.read()

prompt = f"Using this JSON file {json_str} create a python cadquery code that allows me to create the STP model using the JSON instructions"

chat = create_chat()

response = chat.send_message(prompt)

pythonCode = get_pure_python(response.text)

print (pythonCode)




