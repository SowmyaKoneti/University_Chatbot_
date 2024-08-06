from flask import Flask, render_template, request, jsonify
from chatbot import get_response
import traceback
from time import strftime

from flask import Flask, request, jsonify, make_response, render_template

from config.cfg_handler import CfgHandler
from config.cfg_utils import fetch_base_url
from framework.justext.core import justextHTML
from framework.parser.parser import Parser
from implementation import word_frequency_summarize_parser
import json
from flask import Flask, render_template, request, jsonify, make_response
import traceback
import json
from chatbot import get_response
from implementation import word_frequency_summarize_parser
from framework.justext.core import justextHTML
from framework.parser.parser import Parser
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from chatbot import get_conversational_chain

app = Flask(__name__)

# Initialize embeddings and vector database
embeddings1 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb1 = Chroma(persist_directory="data/Embeddings", embedding_function=embeddings1)

# Route for the home page
@app.route("/")
def index_get():
    return render_template("base.html")
# Route for the links page
@app.route("/links")
def links():
    return render_template("links.html")
@app.route('/v1/summarize', methods=['GET'])
def summarize():
    app.logger.debug('summarize(): requested')
    if 'url' not in request.args:
        return make_response(jsonify({'error': str('Bad Request: argument `url` is not available')}), 400)

    url = request.args['url']

    if not url:  # if url is empty
        return make_response(jsonify({'error': str('Bad Request: `url` is empty')}), 400)

    summary = ""

    try:
        # Fetch web content
        web_text = justextHTML(html_text=None, web_url=url)

        # Parse it via parser
        parser = Parser()
        parser.feed(web_text)

        # summary = facebook_parser_word_frequency_summarize.run_summarization(parser.paragraphs)
        summary = word_frequency_summarize_parser.run_summarization(parser.paragraphs)

    except Exception as ex:
        app.logger.error('summarize(): error while summarizing: ' + str(ex) + '\n' + traceback.format_exc())
        pass

    return make_response(jsonify({'summary': summary}))

# Load schemas from a JSON file
with open('schemas.json', 'r') as f:
    schemas = json.load(f)
# Function to get the next questions based on the response
def get_next_questions(response):
    for item in schemas["schemas"]:
        if "responses" in item and response in item["responses"]:
            return item.get("next_questions", [])
        elif "subcategories" in item:
            for subcategory in item["subcategories"]:
                if "responses" in subcategory and response in subcategory["responses"]:
                    return subcategory.get("next_questions", [])
    return []
# Route for handling predictions
@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    # print(response)
    next_questions = get_next_questions(response)
    print(next_questions)
    message = {"answer": response, "next_questions": next_questions}
    return jsonify(message)


@app.route('/job_recommendation', methods=['POST'])
def job_recommendation():
    data = request.get_json()
    question = data.get('message')
    if not question:
        return make_response(jsonify({'error': 'No input provided'}), 400)

    try:
        response = get_conversational_chain(vectordb1, question)
        return jsonify({'answer': response})
    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)

if __name__ == "__main__":
    app.run(debug=True)
