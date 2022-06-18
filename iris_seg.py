import cv2 as cv
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# LEFT_IRIS = [474, 475, 476, 477]
# RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474]
RIGHT_IRIS = [469]
cap = cv.VideoCapture(0)

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=1, min_tracking_confidence=0.5, refine_landmarks=True)


def main(frame):
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        # print(results.multi_face_landmarks[0].landmark)
        mesh_points = np.array(
            [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        # print(mesh_points.shape)

        # draw lines
        for i in range(16):
            cv.line(frame, mesh_points[RIGHT_EYE[i - 1]], mesh_points[RIGHT_EYE[i]], (0, 255, 0), 1)
        for i in range(16):
            cv.line(frame, mesh_points[LEFT_EYE[i - 1]], mesh_points[LEFT_EYE[i]], (0, 255, 0), 1)

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_hsv = np.array([50, 100, 100])
        upper_hsv = np.array([70, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)

        # Bitwise-AND mask and original image
        res = cv.bitwise_and(frame, frame, mask=mask)
        ret, thresh = cv.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, 1, 2)
        contour_count = len(contours)
        # print(contour_count)
        total = 0
        for i in contours:
            # epsilon = 0.1 * cv.arcLength(i, True)
            # approx = cv.approxPolyDP(i, epsilon, True)
            rect = cv.minAreaRect(i)
            area = cv.contourArea(i)
            _, _, w, h = cv.boundingRect(i)
            total += area
            print("Area :",area)
            box = cv.boxPoints(rect)
            box = np.int0(box)

        Area = total/contour_count

    return box,contour_count,frame,Area,w,h
