from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertModel
import torch
from collections import deque   

class SentimentAnalyzerWithMemory:
    def __init__(self, model_path, memory_size=5, label_map=None):
        self.model, self.tokenizer = self.load_albert(model_path)
        self.label_map = label_map if label_map is not None else {0: "Positive", 1: "Negative"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.memory = deque(maxlen=memory_size)

    def load_albert(self, model_path):
        model = AlbertForSequenceClassification.from_pretrained(model_path)
        tokenizer = AlbertTokenizer.from_pretrained(model_path)
        return model, tokenizer

    def add_to_memory(self, user_prompt):
        self.memory.append(user_prompt)

    def get_memory_context(self):
        return ' '.join(self.memory)

    def predict_sentiment(self, user_prompt):
        self.add_to_memory(user_prompt)
        memory_context = self.get_memory_context()
        inputs = self.tokenizer(memory_context, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        predicted_label = predictions.item()
        return self.label_map.get(predicted_label, 'Unknown')

# Example usage:
model_path = "C:/Users/ISHMO_CT/Downloads/Bloomtech/llmbot/chatbot/albert-base-v2"
analyzer = SentimentAnalyzerWithMemory(model_path, memory_size=5)

user_prompts = [
    "this product is rubbish",
    "I had a great experience",
    "the service was awful",
    "I love this!",
    "would not recommend"
]

for prompt in user_prompts:
    predicted_label = analyzer.predict_sentiment(prompt)
    print(f"User prompt: {prompt}")
    print(f"Predicted label: {predicted_label}")