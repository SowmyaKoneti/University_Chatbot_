# University Chatbot

## Overview

The goal of this project is to create a chatbot that can answer users questions about university. To comprehend and react to user input, the chatbot makes use of neural network modeling and natural language processing (NLP) approaches.

## Table of Contents

- [Overview](#system-overview)
- [Development Process](#development-process)
- [Introduction](#getting-started)
- [Utilization](#usage)
- [Features](#features)


## Overview

The system consists of a Python-based chatbot backend and a Flask web application for the front end. Chatbot uses a neural network model that was trained on a dataset of schemas .

### Technologies Used

- Python
- Flask
- PyTorch
- Natural Language Toolkit (NLTK)

## Development Process

### 1. Preparation of Dataset

The dataset, which is kept in 'schemas.json,' contains a variety of query patterns from users pertaining to university data. Every schema has a tag, responses, and patterns.

### 2. Training Model

The dataset is used to train the neural network model. Tokenization, stemming, and generating a bag-of-words representation for every pattern are steps in the training process.

### 3.Implementation of Flask App

The user interface for communicating with the chatbot is provided by the Flask web application. The frontend consists of JavaScript (app.js), HTML, and CSS.

## Introduction

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:

   ```bash
   python train_model.py
   ```

3. Run the Flask app:

   ```bash
   python app.py
   ```

4. Access the chatbot in your browser at [http://localhost:5000](http://localhost:5000).

## Utilization

Type inquiries about university information to communicate with the chatbot. After processing the input, the chatbot produces appropriate responses.

## Features

- By using predetermined patterns, the chatbot interprets user schemas and responds appropriately.
- By typing inquiries, users can communicate with the chatbot.

---

