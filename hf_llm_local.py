## 1. Hugging Face Transformers (GPU-Accelerated)

# accessing libraries 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# accessing 'gpt2' model
model= 'openai-community/gpt2'

# accessing tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)
model_gpt2 = AutoModelForCausalLM.from_pretrained(model)

# moving the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_gpt2.to(device)

# passing a prompt to the ‘gpt2’ LLM
prompt = "tell me about Portugal?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# generating output
outputs = model_gpt2.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


