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
        scoreboard = cv.threshold(scoreboard, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

        kernel = np.ones((2, 2), np.uint8)
        scoreboard = cv.erode(scoreboard, kernel, iterations=1)

        text = pytesseract.image_to_string(scoreboard)
        print(text)

        cv.imshow('detected_scoreboard', scoreboard)
        cv.waitKey(0)

        count += 1
