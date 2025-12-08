import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

devNumber = torch.cuda.current_device()
print(f"the current device number is {devNumber}")


devName = torch.cuda.get_device_name(devNumber)
print(f"GPU name is {devName}") 

#creating a tensor on cpu

T1=torch.randn(4,4)
print("CPU tensor")
print(T1)

# this will convert a cpu tensor with pinned memory to a CUDA tensor
T2=T1.to(device)
print("CUDA tensor")
print(T2)

# Load Mistral 7B Instruct model
print("\nLoading Mistral 7B Instruct v0.3 model...")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager'
)

print(f"Model loaded successfully on {model.device}")

# Example usage
prompt = "Write a Python function to calculate factorial:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    use_cache=False
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nPrompt: {prompt}")
print(f"Generated:\n{generated_text}")
