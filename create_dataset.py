import json
import random

import cv2 as cv
import numpy as np
import data_aug


diff_thresh = 100000000  # empirical value
img_size = 416
data_aug_factor = 3
p_test = 0.1
p_not_train = 0.3

with open('data/top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json') as json_file:
    data = json.load(json_file)

vc = cv.VideoCapture('data/top-100-shots-rallies-2018-atp-season.mp4')
count = 0
last_p1 = None
last_p2 = None
last_s1 = None
last_s2 = None
last_frame = None

while True:
    ret, frame = vc.read()
    if ret:

        if last_frame is not None:
            sub = cv.subtract(last_frame, frame)
            _sum = sub.sum()
        else:
            _sum = diff_thresh + 1

        dataset = ''
        r = np.random.uniform(0, 1)
        if r > p_not_train:
            dataset = 'train/'
        elif r > p_test:
            dataset = 'valid/'
        else:
            dataset = 'test/'

        if str(count) in data:
            p1 = data[str(count)]['name_1']
            p2 = data[str(count)]['name_2']
            s1 = data[str(count)]['score_1']
            s2 = data[str(count)]['score_2']
            bbox = data[str(count)]['bbox']
            bbox_0 = (bbox[2] + bbox[0]) / (2 * frame.shape[1])
            bbox_1 = (bbox[3] + bbox[1]) / (2 * frame.shape[0])
            bbox_2 = (bbox[2] - bbox[0]) / frame.shape[1]
            bbox_3 = (bbox[3] - bbox[1]) / frame.shape[0]

            if not (p1 == last_p1 and p2 == last_p2 and s1 == last_s1 and s2 == last_s2 and _sum < diff_thresh):
                last_p1, last_p2, last_s1, last_s2 = p1, p2, s1, s2
                last_frame = frame.copy()
                frame = cv.resize(frame, (img_size, img_size))
                for i in range(data_aug_factor):
                    frame, [bbox_0, bbox_1, bbox_2, bbox_3] = data_aug.random_flip(frame, [bbox_0, bbox_1, bbox_2, bbox_3], 0.5, 0.5)
                    frame = data_aug.random_color_distort(frame, 30, 18, sat_vari=0.36)
                    if np.random.uniform(0, 1) > 0.8:
                        frame = cv.GaussianBlur(frame, (3, 3), 0)
                    if np.random.uniform(0, 1) > 0.5:
                        noise = np.random.uniform(0.9, 1.1, frame.shape)
                        frame = np.clip(frame * noise, 0, 255)
                    cv.imwrite('dataset/' + dataset + 'images/' + str(count) + '_' + str(i) + '.jpg', frame)
                    f = open('dataset/' + dataset + 'labels/' + str(count) + '_' + str(i) + ".txt", "w")
                    f.write("0 " + str(bbox_0) + ' ' + str(bbox_1) + ' ' + str(bbox_2) + ' ' + str(bbox_3))
                    f.close()

        else:
            print('nothing at frame ' + str(count))
            if np.random.uniform(0, 1) > 0.9:
                frame = cv.resize(frame, (img_size, img_size))
                cv.imwrite('dataset/' + dataset + 'images/' + 'empty_' + str(count) + '.jpg', frame)
        count += 1
    else:
        break

vc.release()
















