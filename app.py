from flask import Flask, render_template, request, jsonify
from chatbot import chatbot_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_bot_response():
    user_text = request.args.get('msg')
    response = chatbot_response(user_text)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
