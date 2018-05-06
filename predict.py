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
ERROR_THRESHOLD = 0.4


def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    print(return_list)
    return return_list


def response(sentence, show_details=False):
    results = classify(sentence)
    return results


khoavien = []
mahocphan = []
tenhocphan = []
malop = []
stage = ""


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
    global stage
    for ml in malop:
        if string.find(str(ml)) >= 0:
            stage = str(ml)
            return collection.find({"Mã lớp": ml}).limit(1)
    for thp in tenhocphan:
        if string.find(thp) >= 0:
            stage = thp
            return collection.find({"Tên HP": thp}).limit(1)
    for kv in khoavien:
        if string.find(kv) >= 0:
            stage = kv
            return collection.find({"Khoa viện": kv})
    for mhp in mahocphan:
        if string.find(mhp) >= 0:
            stage = mhp
            return collection.find({"Mã HP": mhp}).limit(1)
    return False


logging.basicConfig(filename='example.log', level=logging.INFO)


# số lượng sinh viên trong lớp 671888
def exportRes(data, res):
    if res[0][0] == 'khoa viện':
        print(data['Khoa viện'])
        logging.critical(data['Khoa viện'])
    elif res[0][0] == 'lịch học':
        print(data['Thời gian'] + data['Thứ'] + data['Tuần'])
        logging.critical('Thời gian : ' + data['Thời gian'] + ',Thứ' + data['Thứ'] + ',Tuần' + data['Tuần'])
    elif res[0][0] == 'bắt đầu học':
        print(data['Bắt đầu'])
        logging.critical(data['Bắt đầu'])
    elif res[0][0] == 'kết thúc học':
        print(data['Kết thúc'])
        logging.critical(data['Kết thúc'])
    elif res[0][0] == 'Phòng':
        print(data['địa điểm'])
        logging.critical(data['địa điểm'])
    elif res[0][0] == 'số lượng sinh viên':
        print(data['SL Max'])
        logging.critical(data['SL Max'])
    elif res[0][0] == 'hỏi tên học phần':
        print(data['Tên HP'])
        logging.critical(data['Tên HP'])
    elif res[0][0] == 'hỏi mã học phần':
        print(data['Mã HP'])
        logging.critical(data['Mã HP'])
    elif res[0][0] == 'hỏi mã lớp':
        print(data['Mã lớp'])
        logging.critical(data['Mã lớp'])


def main():
    global stage
    readData()
    try:
        client = MongoClient('localhost', 27017)

        # Get the database
        mydb = client['chatbot']
        collection = mydb.get_collection('course')
        while True:
            question = input('>')
            logging.info(question)

            res = response(question)

            if res:
                for i in intents['intents']:
                    if i['tag'] == res[0][0]:
                        responses = random.choice(i['responses'])
                        logging.critical(responses)
                        print(responses)
            else:
                print('Tôi không hiểu bạn hỏi gì')
                continue

            if res[0][0] == 'liệt kê khoa viện':
                print(21)
            else:
                kq = findKeyword(question + stage, collection)
                if kq:
                    for i in kq:
                        exportRes(i, res)

        client.close()

    except IOError:
        print("Error: cannot connect to mongodb")


if __name__ == "__main__":
    main()
