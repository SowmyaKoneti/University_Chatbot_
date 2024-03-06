import random
import json
import torch

from model import NeuralNet
from nltk_utils import tokenize, list_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('schemas.json', 'r') as json_data:
    schemas = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
tokens = data['tokens']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_schema_and_response(msg):
    sentence = tokenize(msg)
    X = list_of_words(sentence, tokens)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    return tag, prob.item()

def generate_response(schema):
    for schema_data in schemas['schemas']:
        if schema_data['tag'] == schema:
            responses = schema_data['responses']
            return random.choice(responses)
        if 'subcategories' in schema_data:
            for subcategory in schema_data['subcategories']:
                if subcategory['tag'] == schema:
                    responses = subcategory['responses']
                    return random.choice(responses)
    return "I'm not sure how to respond to that."

def get_response(msg):
    schema, confidence = get_schema_and_response(msg)

    if confidence > 0.75:
        response = generate_response(schema)
    else:
        response = "I do not understand, could you please rephrase your question."

    return response
