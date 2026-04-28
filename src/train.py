import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from preprocess import clean_text

df = pd.read_csv('data/synthetic_social_media_data.csv')
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['sentiment']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

pickle.dump(model, open('models/model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))