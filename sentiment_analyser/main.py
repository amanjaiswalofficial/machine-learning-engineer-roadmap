import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(w.strip() for w in open("sentiment_analyser\data\stopwords.txt"))

positive_reviews = BeautifulSoup(open("sentiment_analyser\data\electronics\positive.review").read())
positive_reviews = positive_reviews.findAll("review_text")

negative_reviews = BeautifulSoup(open("sentiment_analyser\data\electronics\\negative.review").read())
negative_reviews = negative_reviews.findAll("review_text")

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []


# generating a word_index_map of each word from the reviews
# such that every word has an index in the word_index_map
# kind of a vocabulary map
for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            
for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # create a big enough array
    for token in tokens:
        i = word_index_map[token] # get index
        x[i] += 1 # increase current count
        
    x = x / x.sum()
    x[-1] = label # set the label as provided
    return x

N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1) # 1 as the label as these are positive reviews
    data[i,:] = xy
    i += 1
    
for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0) # 0 as the label as these are negative reviews
    data[i,:] = xy
    i += 1
    
np.random.shuffle(data)

X = data[:, :-1]
Y = data[:, -1]

X_train = X[:-100,]
Y_train = Y[:-100,]
X_test = X[:-100:,]
Y_test = Y[:-100:,]

model = LogisticRegression()
model.fit(X_train, Y_train)
print("Classification Score: {}".format(model.score(X_test, Y_test)))