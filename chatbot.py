import os
from dotenv import load_dotenv, dotenv_values
import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, list_of_words
from framework.parser.parser import Parser
from framework.justext.core import justextHTML
from word_frequency_summarize_parser import run_summarization

from langchain import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.document_transformers import LongContextReorder
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load JSON file
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

# Initialize the model and load pre-trained weights
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# LangChain API keys and initialization
api_key1 = os.getenv("API_KEY")
groq_api_key = os.getenv("groq_api_key")
os.environ["OPENAI_API_KEY"] = "sk-proj-PtvCwFGTPfrwgtDGQWt5T3BlbkFJl3RO912vwXvqr87S0CVt"

# Initialize embeddings
embeddings1 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vector database
vectordb1 = Chroma(persist_directory="data/Embeddings", embedding_function=embeddings1)

# Initialize LLMs
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key1, temperature=0, verbose=True)
llm1 = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

# Define prompt template
# template = """
# You are a job recommendation chatbot. Your role is to provide job recommendations to the user based on the provided context, which contains information from the job postings. You should only use the information in the context to answer the questions or provide recommendations. Do not use any external information or knowledge beyond what is given in the context.

# Ensure your answer is clear, well-structured, and includes bolding for key points, underlining for emphasis, and newlines for separation of each job. If you don't have enough information from the context to answer, say "Sorry, I don't have enough information to provide a recommendation. If the user input doesn't match within the context of information, please mention like no jobs available"

# Provide job recommendations with detailed information including, but not limited to:
# - Position Title
# - VP Area

# Context:
# {context}

# Human: {human_input}
# Chatbot:
# """

template = """
You are a job recommendation chatbot. Your role is to provide job recommendations to the user based on the provided context, which contains information from the job postings. You should only use the information in the context to answer the questions or provide recommendations. Do not use any external information or knowledge beyond what is given in the context.

Ensure your answer is clear, well-structured, and includes **bolding for key points**, _underlining for emphasis_, and newlines for separation of each job. If you don't have enough information from the context to answer, say "Sorry, I don't have enough information to provide a recommendation." If the user input doesn't match within the context of information, please mention like "No jobs available."

Provide job recommendations with detailed information including, but not limited to:
- **Position Title**
- VP Area
- Context

Context:
{context}

Human: {human_input}
Chatbot:
"""

reordering = LongContextReorder()

# Function to get conversational chain for job recommendations
def get_conversational_chain(vectordb, question):
    retrievers = vectordb.as_retriever()
    ensemble_docs = retrievers.invoke(question)
    reordering_docs = reordering.transform_documents(ensemble_docs)
    prompt = PromptTemplate(input_variables=["human_input", "context"], template=template)
    q_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    result = q_chain.invoke({"input_documents": reordering_docs, "human_input": question}, return_only_outputs=False)
    return result["output_text"]

# Function to predict the schema and confidence of the message
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

# Function to generate a response based on the schema
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
        for schema_data in schemas['schemas']:
            if schema_data['tag'] == schema:
                if 'url' in schema_data:
                    url = schema_data['url']
                    # Summarize the content of the URL
                    summary = summarize_url(url)
                    if summary.strip():  # Check if the summary is not empty
                        return f"Here's a summary of the content at {url}: {summary}"
                    elif 'responses' in schema_data:
                        return random.choice(schema_data['responses'])

                elif 'responses' in schema_data:
                    return random.choice(schema_data['responses'])

            if 'subcategories' in schema_data:
                for subcategory in schema_data['subcategories']:
                    if subcategory['tag'] == schema:
                        # if schema == "job":
                        #     return get_conversational_chain(msg)
                        if 'url' in subcategory:
                            url = subcategory['url']
                            # Summarize the content of the URL
                            summary = summarize_url(url)
                            if summary.strip():  # Check if the summary is not empty
                                return f"Here's a summary of the content at {url}: {summary}"
                            elif 'responses' in subcategory:
                                return random.choice(subcategory['responses'])

                        elif 'responses' in subcategory:
                            return random.choice(subcategory['responses'])

     # Check if the question is related to job recommendations
    if any(keyword in msg.lower() for keyword in ["job", "jobs" "position", "career", "employment", "positions", "assistant jobs"]):
        return get_conversational_chain(vectordb1, msg)

    return "I do not understand, could you please rephrase your question."

def summarize_url(url):
    try:
        # Fetch web content
        web_text = justextHTML(html_text=None, web_url=url)

        # Parse it via parser
        parser = Parser()
        parser.feed(web_text)

        # Run summarization
        summary = run_summarization(parser.paragraphs)

        return summary

    except Exception as ex:
        print("Error summarizing URL:", ex)
        return "Error summarizing the URL"
