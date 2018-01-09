import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import pickle
import os
import numpy as np
import cv2


train_dir = 'images/train'
test_dir = 'images/test'
training_data = []
for pic in os.listdir(train_dir):
    if pic.split('_')[0] == 'nick':
        label = np.array([1,0])
    elif pic.split('_')[0] == 'britt':
        label = np.array([0,1])
    img = cv2.imread(os.path.join(train_dir,pic),cv2.IMREAD_GRAYSCALE)
    data = np.array(img)
    training_data.append([data,label])
np.random.shuffle(training_data)

cnn = input_data(shape=[None, 300, 300, 1], name='input')
cnn = conv_2d(cnn, 32, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 64, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 128, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 64, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 32, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = fully_connected(cnn, 1024, activation='relu')
cnn = dropout(cnn, 0.8)
cnn = fully_connected(cnn, 2, activation='softmax')
cnn = regression(cnn, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn)

train = training_data[:-30]
test = training_data[-30:]
X = np.array([i[0] for i in train]).reshape(-1,300,300,1)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1,300,300,1)
test_y = [i[1] for i in test]
model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True)
model.save('model.tflearn')
