from flask import Flask, request, jsonify
from flask_cors import CORS

from classifier import Classifier

app = Flask(__name__)
CORS(app, origins="*")

classifier = Classifier()

@app.route("/", methods=["POST"])
def predict():
    
    global classifier

    payload = request.get_json()
    text = payload.get("input")

    label = classifier.predict_input(text)

    return jsonify({"input": text, "label": label}), 200


if __name__ == "__main__":
    app.run(debug=True)
