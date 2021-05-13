import detect
import cv2 as cv
import pytesseract
import numpy as np
import json
import re
from Levenshtein import distance as levenshtein_distance


def ocr_preprocessing(scoreboard):
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

    # split the image in two rows of text
    first = binary[0:binary.shape[0] // 2, 0:binary.shape[1] - 1]
    second = binary[binary.shape[0] // 2:binary.shape[0] - 1, 0:binary.shape[1] - 1]

    if show_images:
        cv.imshow('binary', binary)
        cv.imshow('first', first)
        cv.imshow('second', second)
    return first, second


def ocr(image):
    height = image.shape[0]
    width = image.shape[1]

    # first = cv.copyMakeBorder(image, 20, 20, 50, 50, cv.BORDER_CONSTANT)  # to add padding

    d = pytesseract.image_to_boxes(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['char'])
    text_composed = ''
    last_center = -1
    last_width = -1
    for i in range(n_boxes):
        (text, x1, y2, x2, y1) = (d['char'][i], d['left'][i], d['top'][i], d['right'][i], d['bottom'][i])
        cv.rectangle(image, (x1, height - y1), (x2, height - y2), (0, 255, 0), 1)

        center = (x1 + x2) / 2
        width = x2 - x1

        if last_width > 0:
            if center - last_center > 1.5 * max(width, last_width):
                text_composed += ' '
        text_composed += text

        last_width = width
        last_center = center
        # print(text)

    return text_composed


def parse_ocr_output(text1, text2):
    player_regex = '[^A-Za-z. ]+'  # this regex will cut out all special char and digits, but point and space
    score_regex = '[^adAD0-9]+'  # this regex will cut out all special char and digits, but point and space
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

    player_1 = re.sub(player_regex, '', text1[:split_index_1])
    player_2 = re.sub(player_regex, '', text2[:split_index_2])

    score_1 = re.sub(score_regex, ' ', text1[split_index_1:])
    score_2 = re.sub(score_regex, ' ', text2[split_index_2:])
    score_1 = '-'.join(score_1.strip().split(' '))  # take the 2nd part->strip->split->join with -
    score_2 = '-'.join(score_2.strip().split(' '))

    print('player1: ' + player_1)
    print('player2: ' + player_2)
    print('score1: ' + score_1)
    print('score2: ' + score_2)
    return player_1, player_2, score_1, score_2


def detect_serving_player(text1, text2, scoreboard, serving_crop_width):
    serving = ''
    if text1.startswith('>') or text1.startswith('»'):  # indicator is often recognized as these char
        serving = 'name_1'
    elif text2.startswith('>') or text2.startswith('»'):
        serving = 'name_2'
    else:  # extract a little portion at the beginning of the lines, choose the brightest one
        first_crop = scoreboard[0:scoreboard.shape[0] // 2, 0:serving_crop_width]
        second_crop = scoreboard[scoreboard.shape[0] // 2:scoreboard.shape[0] - 1, 0:serving_crop_width]
        sum1 = first_crop.sum()
        sum2 = second_crop.sum()
        serving = 'name_1' if sum1 > sum2 else 'name_2'
        if show_images:
            cv.imshow('first_crop', first_crop)
            cv.imshow('second_crop', second_crop)
    print('serving: ' + serving)
    return serving


if __name__ == '__main__':

    show_images = False

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    cap = cv.VideoCapture('../data/top-100-shots-rallies-2018-atp-season.mp4')
    with open('../data/top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json') as json_file:
        data = json.load(json_file)
    count = 0
    serving_crop_width = 10  # the width of the crop to extract before the name to choose who is serving
    model = detect.load_model()
    skip_empty_frames = True
    avg_ocr_error = 0
    frames_with_text = 0
    serving_accuracy = 0

    last_p1, last_p2, last_s1, last_s2 = None, None, None, None  # to avoid visualizing every similar frame

    while cap.isOpened():
        ret, frame = cap.read()

        try:
            if ret:
                if show_images:
                    cv.imshow('original_frame', frame)
                if str(count) in data:
                    sp = data[str(count)]['serving_player']
                    p1 = data[str(count)]['name_1']
                    p2 = data[str(count)]['name_2']
                    s1 = data[str(count)]['score_1']
                    s2 = data[str(count)]['score_2']

                    if p1 != last_p1 or p2 != last_p2 or s1 != last_s1 or s2 != last_s2:
                        last_p1, last_p2, last_s1, last_s2 = p1, p2, s1, s2
                        scoreboard = detect.detect_score_board(model, frame)
                        if show_images:
                            cv.imshow('detected_scoreboard', scoreboard)

                        if scoreboard is not None:
                            frames_with_text += 1
                            first, second = ocr_preprocessing(scoreboard)
                            text1 = ocr(first)
                            text2 = ocr(second)
                            player_1, player_2, score_1, score_2 = parse_ocr_output(text1, text2)

                            serving = detect_serving_player(text1, text2, scoreboard, serving_crop_width)

                            # evaluation
                            avg_ocr_error += levenshtein_distance(
                                (player_1.strip() + player_2.strip() + score_1.strip() + score_2.strip()).upper(),
                                (p1.strip() + p2.strip() + s1.strip() + s2.strip()).upper())
                            if serving == sp:
                                serving_accuracy += 1
                            if show_images:
                                cv.waitKey(0)
                else:
                    if not skip_empty_frames and detect.detect_score_board(model, frame) is not None:
                        print('found a fake scoreboard where there is nothing')
                print(count)
            else:
                break
        except:
            if show_images:
                cv.imshow('detected_scoreboard', scoreboard)
            print('unknown error')
            cv.waitKey(0)
        count += 1

    avg_ocr_error /= frames_with_text
    serving_accuracy /= frames_with_text

    print('Average OCR error: ' + str(avg_ocr_error))
    print('Serving player accuracy: ' + str(serving_accuracy))
