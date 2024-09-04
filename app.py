from flask import Flask, render_template, request, jsonify
from model import conversational_chain

app = Flask(__name__)
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    return get_Chat_response(msg)

def get_Chat_response(query):
    result = conversational_chain({"question" : query})
    return result['answer']

if __name__ == '__main__':
    app.run()