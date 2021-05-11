import detect
import cv2 as cv
import pytesseract
import numpy as np

if __name__ == '__main__':

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    cap = cv.VideoCapture('../data/top-100-shots-rallies-2018-atp-season.mp4')
    count = 0
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

        kernel = np.ones((2, 2), np.uint8)
        binary = cv.erode(binary, kernel, iterations=1)

        text = pytesseract.image_to_string(binary)
        print(text)

        cv.imshow('binary', binary)
        cv.imshow('detected_scoreboard', scoreboard)
        cv.waitKey(0)

        count += 1
