import numpy as np
import random
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import list_of_words, tokenize, stem
from model import NeuralNet

with open('schemas.json', 'r') as f:
    schemas = json.load(f)

tokens = []
tags = []
pattern_tag = []
# loop through each sentence in our schemas patterns
for schema in schemas['schemas']:
    tag = schema['tag']
    # appending to tag 
    tags.append(tag)

    if 'patterns' in schema:
        for pattern in schema['patterns']:
            # tokenize each word 
            w = tokenize(pattern)
            # append to our words 
            tokens.extend(w)
            # append to pattern_tag 
            pattern_tag.append((w, tag))

    if 'subcategories' in schema:
        for subcategory in schema['subcategories']:
            subtag = subcategory['tag']
            tags.append(subtag)

            if 'patterns' in subcategory:
                for pattern in subcategory['patterns']:
                    w = tokenize(pattern)
                    tokens.extend(w)
                    pattern_tag.append((w, subtag))

# stemming and lowering every word
words_ignored = ['?', '.', '!']
tokens = [stem(w) for w in tokens if w not in words_ignored]
tokens = sorted(set(tokens))
tags = sorted(set(tags))

print(len(pattern_tag), "patterns")
print(len(tags), "tags:", tags)
print(len(tokens), "unique stemmed words:", tokens)

# creating training data
X_train = []
y_train = []
for (sentence, tag) in pattern_tag:
    bag = list_of_words(sentence, tokens)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-params 
epochs= 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class Dataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # dataset[i] to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

dataset = Dataset()
train_data_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optim
reference = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(epochs):
    for (words, labels) in train_data_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # Forward pass
        results = model(words)
        loss = reference(results, labels)
        # Backward and optimize
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


print(f'total loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"tokens": tokens,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print('training completed.')
