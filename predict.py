import pickle
import json
import tflearn
from pyvi import ViTokenizer, ViPosTagger
import numpy as np
import random

data = pickle.load(open("models/training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

with open('data/intents.json') as json_data:
    intents = json.load(json_data)

# load saved model
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load('./models/model.tflearn')


def clean_up_sentence(sentence):
    sentence_words = ViPosTagger.postagging(sentence.lower())[0]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the sentence
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag %s" % w)
    return (np.array(bag))


# data structure to hold user context
ERROR_THRESHOLD = 0.1

def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list



def response(sentence,show_details=False):
    results = classify(sentence)
    while results:
        for i in intents['intents']:
            if i['tag']==results[0][0]:
                return print(random.choice(i['responses']))

print(classify('Bao giờ học'))
response('Bao giờ học')
# def program :
    