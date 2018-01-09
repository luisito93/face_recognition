import cv2
import numpy as np
import os
britt_counter = 0
nick_counter = 0
for pic in os.listdir('images/face_images'):
    label = pic.split('_')[0]
    img = cv2.imread(os.path.join('images/face_images/',pic))
    for blur in (True,False):
        for flip in (True,False):
            if blur == True and flip == True:
                img = np.fliplr(img)
                img = cv2.GaussianBlur(img, (5,5), 0)
                if label == 'britt':
                    cv2.imwrite('images/train/britt_%s.jpg'%britt_counter,img)
                    britt_counter +=1
                elif label == 'nick':
                    cv2.imwrite('images/train/nick_%s.jpg'%nick_counter,img)
                    nick_counter +=1
            if blur == True and flip == False :
                img = cv2.GaussianBlur(img, (5,5), 0)
                if label == 'britt':
                    cv2.imwrite('images/train/britt_%s.jpg'%britt_counter,img)
                    britt_counter +=1
                elif label == 'nick':
                    cv2.imwrite('images/train/nick_%s.jpg'%nick_counter,img)
                    nick_counter +=1
            if blur == False  and flip == True:
                img = np.fliplr(img)
                if label == 'britt':
                    cv2.imwrite('images/train/britt_%s.jpg'%britt_counter,img)
                    britt_counter +=1
                elif label == 'nick':
                    cv2.imwrite('images/train/nick_%s.jpg'%nick_counter,img)
                    nick_counter +=1

            if blur == False and flip == False:
                if label == 'britt':
                    cv2.imwrite('images/train/britt_%s.jpg'%britt_counter,img)
                    britt_counter +=1
                elif label == 'nick':
                    cv2.imwrite('images/train/nick_%s.jpg'%nick_counter,img)
                    nick_counter +=1
