import numpy as np
import cv2 as cv




def random_color_distort(img, brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    '''
    randomly distort image color. Adjust brightness, hue, saturation, value.
    param:
        img: a BGR uint8 format OpenCV image. HWC format.
    '''


    def random_hue(img_hsv, hue_vari):
        hue_delta = np.random.randint(-hue_vari, hue_vari)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari):
        sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
        img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari):
        val_mult = 1 + np.random.uniform(-val_vari, val_vari)
        img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta):
        img = img.astype(np.float32)
        brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
        img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

    return img


def random_flip(img, bbox, px=0, py=0):
    '''
    Randomly flip the image and correct the bbox.
    param:
    px:
        the probability of horizontal flip
    py:
        the probability of vertical flip
    '''

    if np.random.uniform(0, 1) < px:
        img = cv.flip(img, 1)
        bbox[0] = 1 - bbox[0]

    if np.random.uniform(0, 1) < py:
        img = cv.flip(img, 0)
        bbox[1] = 1 - bbox[1]
    return img, bbox
