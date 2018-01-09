import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import pickle
import os
import numpy as np
import cv2

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
model.load("model.tflearn")


cap = cv2.VideoCapture('videos/input1.mov')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
out = cv2.VideoWriter('videos/output1.mov',cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), 20, (720,1280))
record_2 = False
for i in range(300):
        _, img = cap.read()
        try:
            faces = face_cascade.detectMultiScale(img, 1.05, 4,minSize=(200,200))
            if len(faces) == 1:
                (x,y,w,h) = faces[0]
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                img_pred = img[y:y+h,x:x+w,:]
                img_pred = cv2.resize(img_pred,(300,300), interpolation = cv2.INTER_CUBIC)
                img_pred = cv2.cvtColor(img_pred,cv2.COLOR_BGR2GRAY)
                pred = np.argmax(model.predict(np.array(img_pred).reshape(-1,300,300,1)))
                preds =  model.predict(np.array(img_pred).reshape(-1,300,300,1))[0]
                nick = round(preds[0],2)
                britt = round(preds[1],2)
                if pred == 0:
                    lab = 'Nick : %s'%nick
                if pred == 1:
                    lab = 'Brittany : %s'%britt
                cv2.putText(img,lab, (x-100,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255,3)

            if len(faces) == 2 and record_2 == True:
                (x1,y1,w1,h1) = faces[0]
                cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
                img_pred = img[y1:y1+h1,x1:x1+w1,:]
                img_pred = cv2.resize(img_pred,(300,300), interpolation = cv2.INTER_CUBIC)
                img_pred = cv2.cvtColor(img_pred,cv2.COLOR_BGR2GRAY)
                pred = np.argmax(model.predict(np.array(img_pred).reshape(-1,300,300,1)))
                preds =  model.predict(np.array(img_pred).reshape(-1,300,300,1))[0]
                nick = round(preds[0],2)
                britt = round(preds[1],2)
                if pred == 0:
                    lab = 'Brittany: %s'%nick
                if pred == 1:
                    lab = 'Nick: %s'%britt
                cv2.putText(img,lab, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, 255,3)

                (x2,y2,w2,h2) = faces[1]
                cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
                img_pred = img[y2:y2+h2,x2:x2+w2,:]
                img_pred = cv2.resize(img_pred,(300,300), interpolation = cv2.INTER_CUBIC)
                img_pred = cv2.cvtColor(img_pred,cv2.COLOR_BGR2GRAY)
                pred = np.argmax(model.predict(np.array(img_pred).reshape(-1,300,300,1)))
                preds =  model.predict(np.array(img_pred).reshape(-1,300,300,1))[0]
                nick = round(preds[0],2)
                britt = round(preds[1],2)
                if pred == 0:
                    lab = 'Brittany: %s'%nick
                if pred == 1:
                    lab = 'Nick: %s'%britt
                cv2.putText(img,lab, (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 2, 255,3)
        except:
            pass
        out.write(img)



cap.release()
out.release()
