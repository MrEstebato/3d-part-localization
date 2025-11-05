from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "ricemonster/codegpt-small-sft"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
# Use 'torch_dtype=torch.float16' and move to 'cuda' if you have a compatible GPU for better performance
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define your input prompt
prompt = "This is a test"

# Encode the prompt
inputs = tokenizer(prompt, return_tensors="pt")
# If using a GPU:
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate text (code)
# You can adjust parameters like max_new_tokens, num_return_sequences, temperature, etc.
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=50,  # Limits the length of the generated code
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,  # Set the end-of-sequence token as the pad token
)

# Decode the generated output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the result
print(generated_text)
