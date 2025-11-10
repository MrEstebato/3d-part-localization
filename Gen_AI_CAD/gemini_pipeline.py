import google.generativeai as genai
import os
import subprocess
import argparse
from prompts import PROMPT
from datetime import datetime
import time


# ===============================
# CONFIGURACIÓN DEL CLIENTE GEMINI
# ===============================
genai.configure(api_key="llave")

# Crea una función para generar modelos/chat
def create_chat():
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=PROMPT,
        generation_config={"temperature": 0}
    )
    return model.start_chat()


# ===============================
# FUNCIONES AUXILIARES
# ===============================
def get_input(json_path, file_index, save_path):
    """Convierte un archivo JSON en un prompt para Gemini."""
    with open(json_path, "r", encoding="utf-8") as f:
        json_str = f.read()
    return (
        f"'''Give me CAD query from this CAD sequence: {json_str}. "
        f"The export file name should be {os.path.join(save_path, file_index)}.stl. "
        "In the end, only save stl file, don't need to use show().'''"
    )


def get_pure_python(gemini_output):
    """Extrae el código Python puro de la respuesta de Gemini."""
    if gemini_output.startswith("```python"):
        gemini_output = gemini_output[len("```python"):].strip()
    if gemini_output.endswith("```"):
        gemini_output = gemini_output[:-len("```")].strip()
    return gemini_output


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Request timed out")


def generate_code_with_timeout(chat, user_input, timeout=60):
    """Envía un prompt al modelo con un límite de tiempo."""

    try:
        response = chat.send_message(user_input)
        result = get_pure_python(response.text)
        return result
    except TimeoutException:
        print(f"Generation timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Error generating code: {e}")
        return None


def generate_and_validate_code(chat, json_str, file_index, cq_dir, max_attempts=None):
    """Genera código CadQuery hasta que funcione correctamente o se alcance el límite de intentos."""
    attempt = 0
    success = False
    cq_path = os.path.join(cq_dir, f"{file_index}.py")

    response = generate_code_with_timeout(chat, json_str)

    while True:
        attempt += 1
        print(f"Intento #{attempt} para {file_index}.py")

        if response is None:
            print("No se recibió respuesta del modelo. Reintentando...")
            response = generate_code_with_timeout(chat, json_str)
            time.sleep(3)
            continue

        # Guarda el código generado
        with open(cq_path, "w", encoding="utf-8") as f:
            f.write(response)

        # Ejecuta el archivo y valida si CadQuery funciona correctamente
        result = subprocess.run(["python", cq_path], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"{file_index}.py ejecutado correctamente.")
            success = True
            break
        else:
            print(f"Error en ejecución de {file_index}.py")
            error_msg = "\n".join(result.stderr.splitlines()[-10:])
            print(f"Últimos errores:\n{error_msg}\n")

            retry_prompt = (
                f"The previous CadQuery code had this error:\n{error_msg}\n"
                "Generate a corrected version of the CadQuery code that works. "
                "Only output valid Python code, without comments or explanations."
            )
            response = generate_code_with_timeout(chat, retry_prompt)

            # Si hay límite de intentos, verifica
            if max_attempts and attempt >= max_attempts:
                print(f"Se alcanzó el número máximo de intentos ({max_attempts}).")
                break

            time.sleep(3)  # Espera breve antes del siguiente intento

    return success, cq_path


# ===============================
# FUNCIÓN PRINCIPAL
# ===============================
def main():
    success_count = 0
    fail_count = 0
    failed_scripts = []

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", type=str, required=True)
    parser.add_argument("-s", "--stl_dir", type=str, required=True)
    parser.add_argument("-c", "--cq_dir", type=str, required=True)
    args = parser.parse_args()

    root_dir = args.root_dir
    stl_dir = args.stl_dir
    cq_dir = args.cq_dir

    os.makedirs("./failed_scripts", exist_ok=True)

    for file in os.listdir(root_dir):
        if file.endswith("json"):
            print(f"Working with {file}")
            file_path = os.path.join(root_dir, file)
            file_index = file.split("_")[0]
            json_str = get_input(file_path, file_index, stl_dir)

            chat = create_chat()

            success, path = generate_and_validate_code(chat, json_str, file_index, cq_dir)
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_scripts.append(path)

    print(f"\nNumber of success: {success_count}")
    print(f"Number of failure: {fail_count}")

    with open(
        os.path.join("./failed_scripts/", f"{os.path.basename(cq_dir)}_failed_scripts.txt"), "w"
    ) as f:
        for item in failed_scripts:
            f.write(str(item) + "\n")


if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print(f"Start: {start.strftime('%H:%M:%S')}")
    print(f"End:   {end.strftime('%H:%M:%S')}")
    print(f"Total execution time: {end - start}")
