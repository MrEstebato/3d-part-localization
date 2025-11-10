import subprocess

# Ejecutar el archivo JSONGemini.py
subprocess.run(["python", "JSONGemini.py"])
print("JSON file created successfully.")

# Ejecutar el archivo CADGemini.py
subprocess.run(["python", "CADGemini.py"])
print("Python file with Cadquery code created successfully.")

# Ejecutar el archivo output.py dónde está el código de CadQuery generado 
subprocess.run(["python", "output.py"])
print("CAD file created successfully.")