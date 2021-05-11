import detect
import cv2 as cv
import pytesseract
import numpy as np

if __name__ == '__main__':

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    cap = cv.VideoCapture('../data/top-100-shots-rallies-2018-atp-season.mp4')
    count = 0
    serving_crop_witdh = 10
    model = detect.load_model()

    while cap.isOpened():
        ret, frame = cap.read()
        original_frame = frame.copy()

        scoreboard = detect.detect_score_board(model, original_frame)

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
        print(text1)
        text2 = pytesseract.image_to_string(second)
        print(text2)

        # EXTRACT THE TWO PLAYER NAMES AND THEIR SCORES
        player_1 = ''
        player_2 = ''
        score_1 = ''
        score_2 = ''

        # WHO IS SERVING?
        serving = ''
        if text1.startswith('>') or text1.startswith('»'):
            serving = player_1
        elif text2.startswith('>') or text2.startswith('»'):
            serving = player_2
        else:  # extract a little portion at the beginning of the lines, choose the brightest one
            first_crop = scoreboard[0:scoreboard.shape[0]//2, 0:serving_crop_witdh]
            second_crop = scoreboard[scoreboard.shape[0]//2:scoreboard.shape[0]-1, 0:serving_crop_witdh]
            sum1 = first_crop.sum()
            sum2 = second_crop.sum()
            serving = player_1 if sum1 > sum2 else player_2

        cv.imshow('binary', binary)
        cv.imshow('first', first)
        cv.imshow('second', second)
        cv.imshow('first_crop', first_crop)
        cv.imshow('second_crop', second_crop)
        cv.imshow('detected_scoreboard', scoreboard)
        cv.waitKey(0)

        count += 1
