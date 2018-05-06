import pickle
import json
from _threading_local import local

import tflearn
from pyvi import ViTokenizer, ViPosTagger
import numpy as np
import random
import logging
import pymongo
from pymongo import MongoClient

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
    sentence_words = ViPosTagger.postagging(ViTokenizer.tokenize(sentence))[0]
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
ERROR_THRESHOLD = 0.2


def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


def response(sentence, show_details=False):
    results = classify(sentence)

    return results


khoavien = []
mahocphan = []
tenhocphan = []
malop = []


def readData():
    with open('data/data.json') as json_data:
        mang = json.load(json_data)

    for i in mang['features']:
        global khoavien
        global malop
        global tenhocphan
        global mahocphan
        khoavien.append(i['Khoa viện'])
        mahocphan.append(i['Mã HP'])
        tenhocphan.append(i['Tên HP'])
        malop.append(i['Mã lớp'])

    khoavien = set(khoavien)
    mahocphan = set(mahocphan)
    tenhocphan = set(tenhocphan)
    malop = set(malop)


def findKeyword(string, collection):
    for ml in malop:
        if string.find(str(ml)) >= 0:
            return collection.find({"Mã lớp": ml}).limit(1)
    for thp in tenhocphan:
        if string.find(thp) >= 0:
            return collection.find({"Tên HP": thp}).limit(1)
    for kv in khoavien:
        if string.find(kv) >= 0:
            return collection.find({"Khoa viện": kv}).limit(1)
    for mhp in mahocphan:
        if string.find(mhp) >= 0:
            return collection.find({"Mã HP": mhp}).limit(1)
    return False


# số lượng sinh viên trong lớp 671888
def exportRes(data, res):
    logging.basicConfig(filename='example.log', level=logging.INFO)

    for i in intents['intents']:
        if i['tag'] == res[0][0]:
            responses = random.choice(i['responses'])
            logging.critical(responses)
            print(responses)

    if res[0][0] == 'khoa viện':
        print(data['Khoa viện'])
    elif res[0][0] == 'lịch học':
        print(data['Thời gian'] + data['Thứ'] + data['Tuần'])
    elif res[0][0] == 'bắt đầu học':
        print(data['Bắt đầu'])
    elif res[0][0] == 'kết thúc học':
        print(data['địa điểm'])
    elif res[0][0] == 'Phòng':
        print(data['địa điểm'])
    elif res[0][0] == 'số lượng sinh viên':
        print(data['SL Max'])
    elif res[0][0] == 'hỏi tên học phần':
        print(data['Tên HP'])
    elif res[0][0] == 'hỏi mã học phần':
        print(data['Mã HP'])
    elif res[0][0] == 'hỏi mã lớp':
        print(data['Mã lớp'])







def main():
    readData()
    try:
        client = MongoClient('localhost', 27017)

        # Get the sampleDB database
        mydb = client['chatbot']
        collection = mydb.get_collection('course')
        while True:
            question = input('>')
            logging.info(question)
            kq = findKeyword(question, collection)
            if kq:
                for i in kq:
                    exportRes(i, response(question))

        client.close()
    except IOError:

        print("Error: cannot connect to rest-mongo-practice")


if __name__ == "__main__":
    main()
