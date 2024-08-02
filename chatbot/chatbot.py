from model_access import tokenizer, model
def chat_with_bot(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["inputs_ids"], max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

user_input = "Hello, how are you?"
response = chat_with_bot(user_input)
