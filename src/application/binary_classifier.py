"""Pipeline for training a binary Logistic Regression classifier to classify job posts per sector (academic vs non academic)"""
import sys

sys.path.append("../jobnet")
import pandas as pd
from preprocessing import identity_tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


df = pd.read_pickle("../../data/pkl/dataset.pkl")
df["occupation_area"].value_counts()

df0 = df[
    (df["occupation_area"] == "pædagogisk socialt og kirkeligt arbejde")
    | (df["occupation_area"] == "akademisk arbejde")
]

posts = df0["lemmas"]

LE = LabelEncoder()
labels = LE.fit(df0["occupation_area"])
name_mapping = dict(zip(LE.classes_, LE.transform(LE.classes_)))
labels = labels.transform(df0["occupation_area"])

x_train, x_test, y_train, y_test = train_test_split(posts, labels, test_size=0.20)

model = LogisticRegression()
tfidf_vectorizer = TfidfVectorizer(
    lowercase=False, tokenizer=identity_tokenizer, ngram_range=(1, 1)
)

X_train = tfidf_vectorizer.fit_transform(x_train)
X_test = tfidf_vectorizer.transform(x_test)

X_train = X_train.toarray()
X_test = X_test.toarray()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

target_names = ["akademisk arbejde", "pædagogisk socialt og kirkeligt arbejde"]
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)
