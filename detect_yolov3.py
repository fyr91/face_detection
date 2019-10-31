# -*- coding: utf-8 -*-
# @Author: User
# @Date:   2019-10-18 14:22:58
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-31 11:54:04
import cv2
import os
import time
import numpy as np

filename = 'crowd3.jpg'
model = 'yolov3-face'
scale = 1

IMG_WIDTH, IMG_HEIGHT = 416, 416
CONFIDENCE = 0.5
THRESH = 0.3

net = cv2.dnn.readNetFromDarknet("Yolo/yolo_models/yolov3-face.cfg", "Yolo/yolo_weights/yolov3-face.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

raw_img = cv2.imread(os.path.join('TestImg',filename))
h, w, _ = raw_img.shape
# img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

# inference
t0 = time.time()

blob = cv2.dnn.blobFromImage(raw_img, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
net.setInput(blob)
layers_names = net.getLayerNames()
outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])

boxes = []
confidences = []
class_ids = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # only face
        if confidence > CONFIDENCE and class_id == 0:
            box = detection[0:4] * np.array([w, h, w, h])
            centerX, centerY, bwidth, bheight = box.astype('int')
            x = int(centerX - (bwidth / 2))
            y = int(centerY - (bheight / 2))

            boxes.append([x, y, int(bwidth), int(bheight)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
# Apply Non-Maxima Suppression to suppress overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESH)

t1 = time.time()
print(f"took {round(t1-t0, 3)} to get {len(idxs.flatten())} faces")

if boxes is None or confidences is None or idxs is None or class_ids is None:
    raise '[ERROR] Required variables are set to None before drawing boxes on images.'

for i in idxs.flatten():
    x, y = boxes[i][0], boxes[i][1]
    w, h = boxes[i][2], boxes[i][3]
    cv2.rectangle(raw_img, (x,y), (x+w,y+h), (80,18,236), 2)

font = cv2.FONT_HERSHEY_DUPLEX
text = f'took {round(t1-t0, 3)} to get {len(idxs.flatten())} faces'
cv2.putText(raw_img, text, (20, 20), font, 0.5, (255, 255, 255), 1)
cv2.imwrite(os.path.join('TestOutput', f'{model}_{scale}_{filename}'), raw_img)
# raw_img = draw_labels_and_boxes(raw_img, boxes, confidences, classids, idxs, colors, labels)

while True:
    cv2.imshow('IMG', raw_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
