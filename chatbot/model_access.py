# Here we are using the tranfoemrs to access the model 
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'meta-llama/LLaMa-3'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)



