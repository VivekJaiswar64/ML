import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Example dataset
data = {
    'text': [
        'Free entry in 2 a wkly comp to win FA Cup final tkts',
        'U dun say so early hor... U c already then say...',
        'Nah I don’t think he goes to usf, he lives around here though',
        'WINNER!! As a valued network customer you have been selected to receive a prize',
        'I HAVE A DATE ON SUNDAY WITH WILL!!',
        'SIX chances to win CASH! From 100 to 20,000 pounds!',
        'I‘m gonna be home soon and i don’t want to talk about this stuff anymore tonight'
    ],
    'label': [1, 0, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# Split and vectorize
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save fitted vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("✅ Trained model and vectorizer saved.")
