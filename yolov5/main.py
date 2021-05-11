import detect
import cv2 as cv

if __name__ == '__main__':

    cap = cv.VideoCapture('../data/top-100-shots-rallies-2018-atp-season.mp4')
    count = 0
    model = detect.load_model()

    while cap.isOpened():
        ret, frame = cap.read()
        original_frame = frame.copy()

        scoreboard = detect.detect_score_board(model, original_frame)

        cv.imshow('detected_scoreboard', scoreboard)
        cv.waitKey(0)

        count += 1
