# -*- coding: utf-8 -*-
# @Author: User
# @Date:   2019-10-16 17:28:22
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-31 11:51:05
import face_recognition
import cv2
import os
import time

filename = 'crowd.jpg'
model = 'hog'
scale = 1

raw_img = cv2.imread(os.path.join('TestImg',filename))
img = raw_img[:, :, ::-1]

t0 = time.time()
print('start')
face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=scale, model=model)
t1 = time.time()
print(f'took {round(t1-t0, 3)} to get {len(face_locations)} faces')

for (top, right, bottom, left) in face_locations:
    cv2.rectangle(raw_img, (left, top), (right, bottom), (80,18,236), 2)

font = cv2.FONT_HERSHEY_DUPLEX
text = f'took {round(t1-t0, 3)} to get {len(face_locations)} faces'
cv2.putText(raw_img, text, (20, 20), font, 0.5, (255, 255, 255), 1)
cv2.imwrite(os.path.join('TestOutput', f'{model}_{scale}_{filename}'), raw_img)

while True:
    cv2.imshow('IMG', raw_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
