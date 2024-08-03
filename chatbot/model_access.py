# Here we are using the tranfoemrs to access the model 
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertModel

model_path = 'C:/Users/ISHMO_CT/Downloads/Bloomtech/llmbot/chatbot/albert-base-v2'

#load the model
model = AlbertForSequenceClassification.from_pretrained(model_path)

#load the tokenizer
tokenizer = AlbertTokenizer.from_pretrained(model_path)  




