import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# with open("data.json", "r") as f: # grammar_chatbot_dataset
with open("grammar_chatbot_dataset.json", "r") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

model = LogisticRegression()
model.fit(X, answers)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved!")
