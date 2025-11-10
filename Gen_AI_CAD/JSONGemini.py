import google.generativeai as genai
import os
import subprocess
import argparse
from prompts import PROMPT
from datetime import datetime
import time

genai.configure(api_key="llave")

def create_chat():
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=PROMPT,
        generation_config={"temperature": 0}
    )
    return model.start_chat()

def get_prompt():
    """ Convierte el prompt a un prompt de Gemini  """
    prompt = input("Bienvenido a tu asistente de creación de CAD, por favor ingresa las modificaciones que quieres hacer:\n")
    
    return prompt


chat = create_chat()

prompt = get_prompt()

response = chat.send_message(prompt)

print (response.text)
