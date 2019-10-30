# -*- coding: utf-8 -*-
# @Author: User
# @Date:   2019-10-17 11:29:48
# @Last Modified by:   User
# @Last Modified time: 2019-10-17 12:33:04
import cv2
from mtcnn.mtcnn import MTCNN
import os
import time

filename = 'crowd3.jpg'
model = 'mtcnn'
scale = 1

detector = MTCNN()
raw_img = cv2.imread(os.path.join('TestImg',filename))
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

t0 = time.time()
print('start')
face_locations = detector.detect_faces(img)
t1 = time.time()
print(f'took {round(t1-t0, 3)} to get {len(face_locations)} faces')

for loc in face_locations:
    (x, y, w, h) = loc['box']
    cv2.rectangle(raw_img, (x,y), (x+w,y+h), (80,18,236), 2)

font = cv2.FONT_HERSHEY_DUPLEX
text = f'took {round(t1-t0, 3)} to get {len(face_locations)} faces'
cv2.putText(raw_img, text, (20, 20), font, 0.5, (255, 255, 255), 1)
cv2.imwrite(os.path.join('TestOutput', f'{model}_{scale}_{filename}'), raw_img)

while True:
    cv2.imshow('IMG', raw_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
