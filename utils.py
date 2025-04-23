import nltk
import numpy as np
import random
import json
import pickle
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess_data(json_file):
    with open(json_file) as file:
        data = json.load(file)

    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']

    for intent in data['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    return words, classes, documents
