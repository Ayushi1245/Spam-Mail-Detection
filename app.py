from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load Model
vectorizer, model = pickle.load(open("model.pkl", "rb"))


@app.route("/", methods=["GET"])
def home():
    return "Flask is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("text")  # Use .get() to avoid errors
    if not data:
        return jsonify({"error": "No text provided"}), 400

    transformed_text = vectorizer.transform([data])
    prediction = model.predict(transformed_text)[0]

    return jsonify({"spam": bool(prediction)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)