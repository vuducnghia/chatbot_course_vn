import pickle
import json
from _threading_local import local

import tflearn
from pyvi import ViTokenizer, ViPosTagger
import numpy as np
import random
import logging

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
    # print(sentence_words)
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
    print(results)
    # if not results:
    #     print("Tôi không hiểu ý của bạn")
    # while results:
    #     for i in intents['intents']:
    #         if i['tag'] == results[0][0]:
    #             res = random.choice(i['responses'])
    #             logging.info(res)
    #             return print(res)
    #     results.pop(0)
    return results


def response_state(sentence, show_details=False):
    results = classify(sentence)
    print(results)
    return results

name = ""
age = ""
address = ""
arrCourse = [" Cửu Âm Chân Kinh", " Càn Khôn Đại Na Di", " Tiên Thiên công", " Cáp Mô Công", " Hàng long thập bát chưởng", " Độc Cô Cửu Kiếm"]


def program():
    courseCurent = ""
    logging.basicConfig(filename='example.log', level=logging.INFO)
    # name = input("Xin Chào! Tôi là trợ lý ảo BOTTOB.Hãy để tôi hỗ trợ các bạn các câu hỏi liên quan đến "
    #              "các khóa học của MYCOURSE\n Bạn tên là gì vậy ?\n")
    # age = input('Bạn bao nhiêu tuổi ?')
    while True:
        question = input('> ')
        question = "  " + question
        logging.info(question)
        if question == "  " :
            print("Bạn chưa điền thông tin ?")
            continue
        if question == "bye":
            response('tạm biệt')

            break
        for i in arrCourse:
            if question.lower().find(i.lower()) > 0:
                courseCurent = i


        # lưu trạng thái nhiều trường hợp data bị lệch dẫn đến kết quả sai => so sánh point giữa có state vs k state
        question_stage = question + courseCurent

        res = response(question)
        res_stage = response_state(question_stage)
        if res[0][1] < res_stage[0][1]:
            # while res_stage:
            for i in intents['intents']:
                if i['tag'] == res_stage[0][0]:
                    responses = random.choice(i['responses'])
                    logging.info(responses)
                    print(responses)
            res_stage.pop(0)
        else:
            # while res:
            for i in intents['intents']:
                if i['tag'] == res[0][0]:
                    responses = random.choice(i['responses'])
                    logging.info(responses)
                    print(responses)
            res.pop(0)
        # response(question)


program()
