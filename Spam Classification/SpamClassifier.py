import pandas as pd

messages = pd.read_csv("./SMSSpamCollection", sep="\t", names=["label", "message"])

# print(messages)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wordnet = WordNetLemmatizer()
corpus = []

for i in range(len(messages)):
    review = re.sub("[^a-zA-Z]", " ", messages["message"][i])
    review = review.lower()
    review = review.split()
    review = [
        wordnet.lemmatize(word)
        for word in review
        if not word in stopwords.words("english")
    ]
    review = " ".join(review)
    corpus.append(review)

# print(corpus)

# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# countvec = CountVectorizer(max_features=5000)
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()
# print(X.shape)
y = pd.get_dummies(messages["label"])
y = y.iloc[:, 1].values
# print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# training model using Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB

spam_detection_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detection_model.predict(X_test)


# evaluation of model
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
