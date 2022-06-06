import mediapipe as mp
import numpy as np
import cv2
from time import time
import itertools
import math
import matplotlib.pyplot as plt
from utils import debug
import utils
debug_bool = True

mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_utils
# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils
face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                                         min_tracking_confidence=0.3)

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]


def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord


# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes
    # cv2.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv2.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio


def detectFacialLandmarks(image, face_mesh, display=True):
    # Perform the facial landmarks detection on the image, after converting it into RGB format.
    results = face_mesh.process(image)
    return results


def getSize(image, face_landmarks, INDEXES):
    image_height, image_width, _ = image.shape
    INDEXES_LIST = list(itertools.chain(*INDEXES))
    landmarks = []
    for INDEX in INDEXES_LIST:
        # Append the landmark into the list.
        landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width),
                          int(face_landmarks.landmark[INDEX].y * image_height)])
    _, _, width, height = cv2.boundingRect(np.array(landmarks))
    landmarks = np.array(landmarks)
    return width, height, landmarks


def isOpen(image, face_mesh_results, face_part, threshold=5, display=True):
    image_height, image_width, _ = image.shape
    output_image = image.copy()
    status = ""
    if face_part == 'MOUTH':
        INDEXES = mp_face_mesh.FACEMESH_LIPS
        loc = (10, image_height - image_height // 40)
        increment = -30
    else:
        return

    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
        _, height, _ = getSize(image, face_landmarks, INDEXES)
        _, face_height, _ = getSize(image, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL)
        # debug(f"threshold : {threshold} mouth ratio : {(height / face_height) * 100}",debug_bool)
        if (height / face_height) * 100 > threshold:
            status = 'Mouth_Open'
            color = (0, 255, 0)
        else:
            status = 'Mouth_Close'
            color = (0, 0, 255)
        cv2.putText(output_image, f'FACE {face_no + 1} {face_part} {status[face_no]}.',
                    (loc[0], loc[1] + (face_no * increment)), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 2)
    return output_image, status,(height / face_height) * 100


def main(frame,eye_frame,mouth_frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    face_mesh_results = detectFacialLandmarks(frame, face_mesh_videos, display=False)
    eye_ratio = 0
    mouth_ratio = 0
    try:
        if face_mesh_results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, face_mesh_results, False)
            eye_ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            left_eye_x = []
            left_eye_y = []
            right_eye_x = []
            right_eye_y = []
            mouth_x = []
            mouth_y = []
            for index in range(len(mesh_coords)):
                if index in LEFT_EYE:
                    left_eye_x.append(mesh_coords[index][0])
                    left_eye_y.append(mesh_coords[index][1])
                if index in RIGHT_EYE:
                    right_eye_x.append(mesh_coords[index][0])
                    right_eye_y.append(mesh_coords[index][1])
                if index in LIPS:
                    mouth_x.append(mesh_coords[index][0])
                    mouth_y.append(mesh_coords[index][1])
            min_x = min(min(left_eye_x),min(right_eye_x))
            min_y = min(min(left_eye_y),min(right_eye_y))
            max_x = max(max(left_eye_x),max(right_eye_x))
            max_y = max(max(left_eye_y),max(right_eye_y))
            offset_eye = 10
            eye_frame = frame[min_y-offset_eye:max_y+offset_eye,min_x-offset_eye:max_x+offset_eye]
            eye_frame = cv2.resize(eye_frame, (300, 150))
            min_x = min(mouth_x)
            min_y = min(mouth_y)
            max_x = max(mouth_x)
            max_y = max(mouth_y)
            offset_mouth = 10
            mouth_frame = frame[min_y - offset_mouth:max_y + offset_mouth, min_x - offset_mouth:max_x + offset_mouth]
            mouth_frame = cv2.resize(mouth_frame, (300, 150))
            frame, status, mouth_ratio = isOpen(frame, face_mesh_results, 'MOUTH', threshold=25, display=False)

            # debug(f"threshold : {4.2} Eye ratio : {eye_ratio}", debug_bool)
            # debug(f"threshold : {25} Mouth ratio : {mouth_ratio}", debug_bool)
        return mouth_ratio, eye_ratio,eye_frame,mouth_frame
    except Exception as e:
        print(f"error : {e}")
        eye_ratio = 0
        mouth_ratio = 0
        return mouth_ratio, eye_frame,mouth_frame
