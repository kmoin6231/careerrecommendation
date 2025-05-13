import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib

df = pd.read_csv("../dataset/career_data.csv")
all_subjects = sorted(set(sum([row[1:] for row in df.values.tolist()], [])))

vectors = []
for row in df.values.tolist():
    subjects = row[1:]
    vector = [1 if subject in subjects else 0 for subject in all_subjects]
    vectors.append(vector)

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(vectors)

joblib.dump({"knn": knn, "all_subjects": all_subjects}, "model.pkl")
print("Model trained and saved.")
