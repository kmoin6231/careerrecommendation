from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

df = pd.read_csv("../dataset/career_data.csv")
model = joblib.load("model.pkl")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    subjects = data.get("subjects", [])

    input_vector = [1 if subject in subjects else 0 for subject in model['all_subjects']]
    distances, indices = model['knn'].kneighbors([input_vector])
    
    recommended = df.iloc[indices[0]][:5]['career'].tolist()
    return jsonify({"recommendations": recommended})

if __name__ == "__main__":
    app.run(debug=True)
