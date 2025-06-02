import pickle
from flask import Flask, request, render_template

# Load trained model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        vector_input = tfidf.transform([text])
        result = model.predict(vector_input)[0]
        return render_template("index.html", prediction=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
