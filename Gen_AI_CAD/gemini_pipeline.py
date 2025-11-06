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
genai.configure(api_key="AIzaSyD1x6DsTlcnCd_B3H5DFpC3f6B0o2-mqvQ")

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


def generate_and_validate_code(chat, json_str, file_index, cq_dir):
    """Genera código CadQuery, lo guarda y lo valida ejecutándolo."""
    response = generate_code_with_timeout(chat, json_str)
    cq_path = os.path.join(cq_dir, f"{file_index}.py")

    if response is None:
        return False, cq_path

    for attempt in range(2):
        with open(cq_path, "w", encoding="utf-8") as f:
            f.write(response)

        result = subprocess.run(["python", cq_path], capture_output=True, text=True)

        if result.returncode == 0:
            return True, cq_path
        else:
            if attempt == 0:
                print(f"⚙️ Error ejecutando {file_index}.py")
                error_msg = "\n".join(result.stderr.splitlines()[-5:])
                retry_prompt = (
                    f"code: {response} has an error: {error_msg}, "
                    "generate it again, only give me python code"
                )
                response = generate_code_with_timeout(chat, retry_prompt)
                if response is None:
                    return False, cq_path
    return False, cq_path


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
            print(f"🧩 Working with {file}")
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
