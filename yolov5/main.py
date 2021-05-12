import detect
import cv2 as cv
import pytesseract
import numpy as np
import json
import re

if __name__ == '__main__':

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    cap = cv.VideoCapture('../data/top-100-shots-rallies-2018-atp-season.mp4')
    with open('../data/top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json') as json_file:
        data = json.load(json_file)
    count = 0
    serving_crop_width = 10  # the width of the crop to extract before the name to choose who is serving
    model = detect.load_model()

    last_p1, last_p2, last_s1, last_s2 = None, None, None, None  # to avoid visualizing every similar frame

    while cap.isOpened():
        ret, frame = cap.read()
        original_frame = frame.copy()
        if ret:
            if str(count) in data:
                p1 = data[str(count)]['name_1']
                p2 = data[str(count)]['name_2']
                s1 = data[str(count)]['score_1']
                s2 = data[str(count)]['score_2']

                if p1 != last_p1 or p2 != last_p2 or s1 != last_s1 or s2 != last_s2:
                    last_p1, last_p2, last_s1, last_s2 = p1, p2, s1, s2
                    scoreboard = detect.detect_score_board(model, original_frame)

                    if scoreboard is not None:
                        # OCR
                        scoreboard = cv.cvtColor(scoreboard, cv.COLOR_BGR2GRAY)
                        # scoreboard = cv.Canny(scoreboard, 100, 200)
                        binary = cv.threshold(scoreboard, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

                        # the highlighted part of the score board is the main source of errors, i try to invert that portion
                        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                        max_area = 0
                        max_area_index = -1
                        for i in range(len(contours)):
                            if max_area < cv.contourArea(contours[i]):
                                max_area = cv.contourArea(contours[i])
                                max_area_index = i

                        cv.drawContours(binary, contours, max_area_index, color=0, thickness=cv.FILLED)

                        first_child_index = hierarchy[0][max_area_index][2]

                        while first_child_index != -1:
                            cv.drawContours(binary, contours, first_child_index, color=255, thickness=cv.FILLED)
                            child = hierarchy[0][first_child_index][2]
                            if child != -1:
                                cv.drawContours(binary, contours, child, color=0, thickness=cv.FILLED)

                            first_child_index = hierarchy[0][first_child_index][0]

                        # binary = cv.resize(binary, (0,0), fx=2, fy=2)
                        kernel = np.ones((2, 2), np.uint8)
                        binary = cv.dilate(binary, kernel, iterations=1)
                        # binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

                        binary = cv.bitwise_not(binary)
                        first = binary[0:binary.shape[0]//2, 0:binary.shape[1]-1]
                        second = binary[binary.shape[0]//2:binary.shape[0]-1, 0:binary.shape[1]-1]

                        text1 = pytesseract.image_to_string(first)
                        # print(text1)
                        text2 = pytesseract.image_to_string(second)
                        # print(text2)

                        # EXTRACT THE TWO PLAYER NAMES AND THEIR SCORES
                        regex = '[^A-Za-z. ]+'  # this regex will cut out all special char and digits, but point and space
                        split_index_1 = -1
                        for i in range(len(text1)):  # i will split the string at the first digit found (weak heuristic)
                            if text1[i].isdigit():
                                split_index_1 = i
                                break
                        split_index_2 = -1
                        for i in range(len(text2)):
                            if text2[i].isdigit():
                                split_index_2 = i
                                break

                        player_1 = re.sub(regex, '', text1[:split_index_1])
                        player_2 = re.sub(regex, '', text2[:split_index_2])
                        score_1 = '-'.join(text1[split_index_1:].strip().split(' '))  # take the 2nd part->strip->split->join with -
                        score_2 = '-'.join(text2[split_index_2:].strip().split(' '))

                        print('player2: ' + player_2)
                        print('player1: ' + player_1)
                        print('score1: ' + score_1)
                        print('score2: ' + score_2)

                        # WHO IS SERVING?
                        serving = ''
                        if text1.startswith('>') or text1.startswith('»'):  # indicator is often recognized as these char
                            serving = player_1
                        elif text2.startswith('>') or text2.startswith('»'):
                            serving = player_2
                        else:  # extract a little portion at the beginning of the lines, choose the brightest one
                            first_crop = scoreboard[0:scoreboard.shape[0]//2, 0:serving_crop_width]
                            second_crop = scoreboard[scoreboard.shape[0]//2:scoreboard.shape[0]-1, 0:serving_crop_width]
                            sum1 = first_crop.sum()
                            sum2 = second_crop.sum()
                            serving = player_1 if sum1 > sum2 else player_2
                            cv.imshow('first_crop', first_crop)
                            cv.imshow('second_crop', second_crop)
                        print('serving: ' + serving)

                        cv.imshow('binary', binary)
                        cv.imshow('first', first)
                        cv.imshow('second', second)

                        cv.imshow('detected_scoreboard', scoreboard)
                        cv.imshow('original_frame', original_frame)
                        cv.waitKey(0)
            else:
                if detect.detect_score_board(model, original_frame) is not None:
                    print('found a fake scoreboard where there is nothing')
            count += 1
