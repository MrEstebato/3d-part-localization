import cadquery as cq
import json

# Load the JSON data
with open("Carpet1.json", "r") as f:
    data = json.load(f)

# Create the CadQuery model based on the JSON data
print(data['metadata'])
