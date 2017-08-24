import cv2
import pickle
import mnist
import numpy as np
from math import log

prior_distribution = pickle.load(open("./data/prior_distribution.p", "rb"))
pixel_probabilities = pickle.load(open("./data/pixel_probabilities.p", "rb"))

def binarize_image(image):
    """
    Applies a fixed threshold on a flattened image. For each pixel x_i, 
    x_i = 1 if x_i >= 1/2 and x_i = 0 otherwise
    @param image A flattened image
    """
    for i in range(0, len(image)):
        if image[i] >= 0.5:
            image[i] = 1
        else:
            image[i] = 0

def estimate_digit(image):
    """
    Given a (28,28) grayscale image, estimates which digit [0,...,9] is written in it
    @param image A (28,28) grayscale image
    @return The digit that is most likely to be the one written in the image
    """
    # image = image.flatten() * 1/255
    # binarize_image(image)
    image = image.flatten()
    best_digit = None
    best_max = None
    
    # h(x) function, but in code
    # Since we are working over a small discrete space [0,...,9], we can simply
    # loop through all 10 digits and see which digit maximizes h(x)
    for j in range(10):
        s = log(prior_distribution[j])
        for i in range(len(image)):
            x_i = image[i]
            p_ji = pixel_probabilities[j][i]
            s+= x_i * log(p_ji) + (1-x_i) * log(1-p_ji)
        if best_max is None or s > best_max:
            best_max = s
            best_digit = j
    return best_digit

def start_scanning():

    cap = cv2.VideoCapture(0)
    while(True):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.rectangle(frame, (440,160), (440 + 400, 160+400), (0, 255, 0), 5)

        roi = gray[160:440, 160+400:440+400]
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized_roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        digit = str(estimate_digit(resized_roi))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, digit, (640, 640), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Scanning...', frame)
        cv2.imshow('ROI', roi)
        cv2.imshow('Resized ROI', resized_roi)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

start_scanning()
