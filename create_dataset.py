import json
import cv2 as cv


with open('data/top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json') as json_file:
    data = json.load(json_file)

vc = cv.VideoCapture('data/top-100-shots-rallies-2018-atp-season.mp4')
count = 0
last_p1 = None
last_p2 = None
last_s1 = None
last_s2 = None

while True:
    ret, frame = vc.read()
    if ret:
        if str(count) in data:
            p1 = data[str(count)]['name_1']
            p2 = data[str(count)]['name_2']
            s1 = data[str(count)]['score_1']
            s2 = data[str(count)]['score_2']
            bbox = data[str(count)]['bbox']
            if not (p1 == last_p1 and p2 == last_p2 and s1 == last_s1 and s2 == last_s2):
                last_p1, last_p2, last_s1, last_s2 = p1, p2, s1, s2

                cv.imshow('scoreboard', frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                print(count)
                cv.waitKey(0)
        else:
            print('nothing at frame ' + str(count))
        count += 1
    else:
        break

vc.release()
















