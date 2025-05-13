import csv
import random
from flask import Flask, render_template, request

app = Flask(__name__)

# Load intents from the updated CSV
def load_intents():
    intents = []
    with open('chat_log.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            intents.append({
                "input": row["input"],
                "response": row["response"]
            })
    return intents

# Function to get bot response
def get_response(user_input, intents):
    for intent in intents:
        if intent['input'].lower() in user_input.lower():
            return random.choice([intent['response']])
    return "I'm sorry, I don't understand that. Can you rephrase?"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chatbot_response():
    user_message = request.args.get('msg')
    intents = load_intents()
    bot_response = get_response(user_message, intents)
    return bot_response

if __name__ == "__main__":
    app.run(debug=True)
