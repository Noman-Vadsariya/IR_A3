from flask import Flask, request, jsonify
from flask_cors import CORS

from classifier import Classifier

app = Flask(__name__)
CORS(app, origins="*")

classifier = Classifier()

@app.route("/train", methods=["POST"])
def train_model():
    
    try:
        global classifier

        label = classifier.train_model()

        return jsonify({"msg": "Model Trained"}), 200

    except:

        return jsonify({"msg": "Error in training model"}), 500


@app.route("/test", methods=["POST"])
def test_model():
    
    try:
        global classifier

        labels , report = classifier.predict_test_data()

        return jsonify({"labels": labels, "report":report}), 200

    except:

        return jsonify({"msg": "Error in testing model"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    
    global classifier

    payload = request.get_json()
    text = payload.get("input")

    classifier.train_model()
    label = classifier.predict_input(text)

    return jsonify({"input": text, "label": label}), 200


if __name__ == "__main__":
    app.run(debug=True)
