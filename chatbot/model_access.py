# Here we are using the tranfoemrs to access the model 
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertModel
def load_albert(model_path):
    model = AlbertForSequenceClassification.from_pretrained(model_path)
    tokenizer = AlbertTokenizer.from_pretrained(model_path)
    return model, tokenizer







