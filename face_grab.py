import numpy as np
import cv2
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
n = 100

for label in ('nick','britt'):
    counter = 0
    for i in range(1,n):
        try:
            img = cv2.imread('images/raw_images/%s%s.jpg'%(label,i))
            faces = face_cascade.detectMultiScale(img, 1.05, 3,minSize=(200,200))
            (x,y,w,h) = faces[0]
            img = img[y:y+h,x:x+w,:]
            img = cv2.resize(img,(300,300), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite("images/face_images/%s_%s.jpg"%(label,counter),img)
            counter += 1
        except:
            print 'No face detected in frame %s'%i
            pass
