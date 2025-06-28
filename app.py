from flask import Flask
from flask import Flask, render_template, request
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
with open("logreg_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    original_text = None
    if request.method == "POST":
        text = request.form["text"]
        original_text = text
        if text.strip() != "":
            processed = preprocess(text)
            vectorized = vectorizer.transform([processed])

            if vectorized.nnz == 0:  # nnz = number of non-zero values
                prediction = "Out of scope"
            else:
                result = model.predict(vectorized)
                prediction = "Fake" if result[0] == 1 else "Real"
    return render_template("index.html", prediction=prediction, original_text=original_text)

if __name__ == "__main__":
    app.run(debug=True, port=10000)
