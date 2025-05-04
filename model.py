import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from preprocessing import preprocess
import nltk

nltk.download('stopwords')


def load_model():
    df = pd.read_csv('../spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['text'] = df['text'].apply(preprocess)

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model, vectorizer
