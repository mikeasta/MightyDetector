# TODO
# 0. DONE Попробовать применить морфологию к адаптивному порогу
# 1. DONE Пофиксить изменение перспективы
# 2.  - Ввести определение на основе ключевых точек

import glob
import math
import typing
import pathlib
import cv2 as cv
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

WINDOW_NAME = "frame"


def load_suits():
    """Load suits templates"""
    # Initiate ORB detector
    orb = cv.ORB_create()

    # For template data
    suit_templates = {}
    suit_keypoints = {}
    suit_descriptors = {}

    # Extract templates
    suit_paths = pathlib.Path("suits").glob("*.jpg")
    for suit_path in suit_paths:
        name = suit_path.name[:-4]
        template = cv.imread(suit_path, cv.IMREAD_GRAYSCALE)
        template = cv.resize(template, (200, 200))
        template = cv.bitwise_not(template)
        ret, template = cv.threshold(template, 100, 255, cv.THRESH_BINARY)
        kp = orb.detect(template, None)
        kp, des = orb.compute(template, kp)
        template = cv.drawKeypoints(template, kp, None, color=(0, 255, 0), flags=0)

        suit_templates[name] = template
        suit_keypoints[name] = kp
        suit_descriptors[name] = des

    return suit_templates, suit_keypoints, suit_descriptors


def main():
    """Main function"""
    prev_tick_count = cv.getTickCount()

    # Load suits
    suit_templates, suit_keypoints, suit_descriptors = load_suits()
    cv.imshow("Playing Card Suits", np.hstack(list(suit_templates.values())))
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Initiate ORB detector and brute-force descriptor matcher
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING)

    cap = cv.VideoCapture(0)
    while cap.isOpened():
        # Load card
        ret, frame = cap.read()

        # Bilateral filter to remove noise
        frame = cv.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # CARD SEARCH
        # Thresholding
        image_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        image_blurred = cv.GaussianBlur(
            src=image_gray,
            ksize=(3, 3),
            sigmaX=0
        )

        # Static threshold for card detection
        ret, binary_thresh = cv.threshold(image_blurred, 180, 255, cv.THRESH_BINARY)

        # Contours search
        contours, hierarchy = cv.findContours(
            image=binary_thresh,
            mode=cv.RETR_EXTERNAL,
            method=cv.CHAIN_APPROX_SIMPLE
        )

        # Remove too little contours
        contours = list(filter(lambda x: cv.contourArea(x) >= 500, contours))

        # Remove non-rectangle contours
        rectangle_contours = []
        for index, cnt in enumerate(contours):
            epsilon = 0.1 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4:
                rectangle_contours.append(cnt)

        cv.drawContours(frame, rectangle_contours, -1, (0, 255, 0), 3)

        # CARD IDENTIFICATION
        # Adaptive thresh for identifying card value and suit
        adaptive_thresh = cv.adaptiveThreshold(
            src=image_gray,
            maxValue=200,
            adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv.THRESH_BINARY,
            blockSize=15,
            C=2
        )

        # Apply erosion for better rank and suit visibility
        cv.morphologyEx(adaptive_thresh, cv.MORPH_ERODE, (9, 9), adaptive_thresh)

        # Value and suit matching
        card_suit_matches = []
        for cnt in rectangle_contours:
            # Define bounding box for matching optimization
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (155, 0, 155), 2)

            # Minimal bounding box coordinates for destination perspective
            dst_pts = np.array([
                [x, y], [x+w, y],
                [x+w, y+h], [x, y+h]
            ], np.float32)

            # Taking approximate corners
            epsilon = 0.1 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)

            # Ordering roi corners clockwise
            corners = approx[:, 0, :].copy()
            roi_pts = np.zeros((4, 2), np.float32)

            s = corners.sum(axis=1)
            roi_pts[0] = corners[np.argmin(s)]
            roi_pts[2] = corners[np.argmax(s)]

            diff = np.diff(corners, axis=1)
            roi_pts[1] = corners[np.argmin(diff)]
            roi_pts[3] = corners[np.argmax(diff)]

            # Perform perspective transform
            mat = cv.getPerspectiveTransform(roi_pts, dst_pts)
            roi = cv.warpPerspective(adaptive_thresh.copy(), mat, (adaptive_thresh.shape[::-1]))[y:y + h, x:x + w]

            # Detect key points and descriptors
            kp = orb.detect(roi, None)
            kp, des = orb.compute(roi, kp)
            roi = cv.drawKeypoints(roi, kp, None, color=(0, 255, 0), flags=0)
            cv.imshow("roi", roi)

            # Suit matching
            best_correct_suit_matches_count = 0
            best_correct_suit_match = None
            for suit_key, suit_des in suit_descriptors.items():
                matches = bf.knnMatch(des, suit_des, k=2)
                correct_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        correct_matches.append(m)

                if len(correct_matches) > best_correct_suit_matches_count:
                    best_correct_suit_matches_count = len(correct_matches)
                    best_correct_suit_match = suit_key

            # Save matching data
            card_suit_matches.append(best_correct_suit_match)

        # Find centroids to mark cards
        for index, cnt in enumerate(rectangle_contours):
            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv.putText(
                frame,
                f"#{index}: {card_suit_matches[index]}",
                (cx-5, cy-5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        # Image comments
        binary_thresh = cv.cvtColor(binary_thresh, cv.COLOR_GRAY2BGR)
        cv.putText(
            binary_thresh,
            "Binary",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        adaptive_thresh = cv.cvtColor(adaptive_thresh, cv.COLOR_GRAY2BGR)
        cv.putText(
            adaptive_thresh,
            "Adaptive",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Log FPS data
        current_tick_count = cv.getTickCount()
        frames_per_second = round(1 / ((current_tick_count - prev_tick_count) / cv.getTickFrequency()))
        prev_tick_count = current_tick_count
        cv.putText(
            frame,
            f"FPS: {frames_per_second}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        result = np.hstack([frame, binary_thresh, adaptive_thresh])
        cv.imshow("card", result)
        if cv.waitKey(20) == ord("s"):
            print("Camera Stopped")
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
