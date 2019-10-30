# -*- coding: utf-8 -*-
# @Author: User
# @Date:   2019-10-17 12:18:50
# @Last Modified by:   User
# @Last Modified time: 2019-10-17 14:24:35
import cv2
from faced import FaceDetector
import os
import time

filename = 'crowd3.jpg'
model = 'faced'
scale = 1

detector = FaceDetector()
raw_img = cv2.imread(os.path.join('TestImg',filename))
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

t0 = time.time()
print('start')
face_locations = detector.predict(img, 0.5)
t1 = time.time()
print(f'took {round(t1-t0, 3)} to get {len(face_locations)} faces')\

for (x, y, w, h, _) in face_locations:
    cv2.rectangle(raw_img, (x-int(w/2),y-int(h/2)), (x+int(w/2),y+int(h/2)), (80,18,236), 2)

font = cv2.FONT_HERSHEY_DUPLEX
text = f'took {round(t1-t0, 3)} to get {len(face_locations)} faces'
cv2.putText(raw_img, text, (20, 20), font, 0.5, (255, 255, 255), 1)
cv2.imwrite(os.path.join('TestOutput', f'{model}_{scale}_{filename}'), raw_img)

while True:
    cv2.imshow('IMG', raw_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
