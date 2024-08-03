from model_access import tokenizer, model
import torch


text = "I love this product!"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

print(f"Predicted class: {predictions.item()}")

# Example label mapping for binary classification
label_map = {
                0: "posiive", 
                1: "Negative"
            }

# Use the mapping to interpret the prediction
predicted_label = predictions.item()
print(f"Predicted class: {label_map.get(predicted_label, 'Unknown')}")

