import nltk
import tflearn
import tensorflow as tf
import numpy as np
import random
import json
from pyvi import ViTokenizer, ViPosTagger
import pickle

with open('data/intents.json') as json_data:
    intents = json.load(json_data)

words = []
words1 = []
classes = []
documents = []
stop_words = ['bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 'chỉ', 'chiếc', 'cho', 'chứ', 'chưa', 'chuyện', 'có',
              'có_thể', 'cứ', 'của', 'cùng', 'cũng', 'đã', 'đang', 'đây', 'để', 'đến nỗi', 'đều', 'điều', 'do', 'đó',
              'được', 'dưới', 'gì', 'khi', 'không', 'là', 'lại', 'lên', 'lúc', 'mà', 'mỗi', 'một_cách', 'này', 'nên',
              'nếu', 'ngay', 'nhiều', 'như', 'nhưng', 'những', 'nơi', 'nữa', 'phải', 'qua', 'ra', 'rằng', 'rất',
              'theo', 'thì', 'sẽ', 'rồi', 'sau', 'tại', 'trên', 'trước', 'từ', 'từng', 'và', 'vẫn', 'vào', 'vậy',
              'vừa', 'với', 'vì', '?', 'à', 'ừ', 'ạ']

# tokenize pattern and response
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = ViPosTagger.postagging(pattern.lower())[0]
        words.extend(w)
        documents.append((w, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

for w in words:
    if w not in stop_words:
        words1.append(w)
words = sorted(list(set(words1)))

# Create training data
training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]

    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists   [ first_row:last_row , column_0 ]
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('models/model.tflearn')

pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y},
            open("models/training_data", "wb"))