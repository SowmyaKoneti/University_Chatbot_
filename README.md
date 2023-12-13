# University Enquiry Chatbot

## Overview

This project aims to develop a chatbot to assist users with queries related to university information. The chatbot utilizes natural language processing (NLP) techniques and a neural network model to understand and respond to user inputs.

## Table of Contents

- [System Overview](#system-overview)
- [Development Process](#development-process)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Features](#features)
- [Challenges Faced](#challenges-faced)
- [Future Enhancements](#future-enhancements)
- [License](#License)

## System Overview

The system comprises a Flask web application as the front end and a Python-based chatbot backend. The chatbot uses a neural network model trained on a dataset of intents.

### Technologies Used

- Python
- Flask
- PyTorch
- Natural Language Toolkit (NLTK)

## Development Process

### 1. Dataset Preparation

The dataset, stored in 'intents.json,' includes various patterns of user queries related to university information. Each intent includes patterns, responses, and a tag.

### 2. Model Training

The neural network model is trained using the dataset. The training process involves tokenization, stemming, and creating a bag-of-words representation for each pattern.

### 3. Flask App Implementation

The Flask web application provides a user interface for interacting with the chatbot. The frontend includes HTML, CSS, and JavaScript (app.js).

### 4. Integration and Testing

The trained model is integrated into the Flask app, allowing users to input questions and receive responses. Extensive testing is conducted to ensure the chatbot's accuracy and responsiveness.

## Getting Started

To run the chatbot locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/oluseyemichael/university-enquiry-chatbot.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:

   ```bash
   python train_model.py
   ```

4. Run the Flask app:

   ```bash
   python app.py
   ```

5. Access the chatbot in your browser at [http://localhost:5000](http://localhost:5000).

## Usage

Interact with the chatbot by typing questions related to university information. The chatbot processes the input and generates appropriate responses.

## Features

- Intent Recognition: The chatbot recognizes user intents and provides relevant responses based on predefined patterns.
- User Interaction: Users can interact with the chatbot by typing questions.

## Challenges Faced

- Limited dataset size during model training.
- Fine-tuning hyperparameters for optimal performance.
- Ensuring seamless communication between the Flask app and the chatbot backend.

## Future Enhancements

1. User Personalization: Implement a user profiling system to personalize responses based on the user's preferences and history.
2. Database Integration: Explore options for integrating a database to store and retrieve dynamic information.

## License

This project is licensed under the [MIT License](LICENSE).

---

