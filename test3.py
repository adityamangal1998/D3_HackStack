from re import X
import cv2
import mediapipe as mp
import math
import numpy as np
import head_main_test
from typing import List, Mapping, Optional, Tuple, Union
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2
import head_main_test_1

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
radius = 2
color = (0, 255, 0)
thickness = -1
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def get_roi_face(image: np.ndarray,
                 detection: detection_pb2.Detection):
    image_rows, image_cols, _ = image.shape

    location = detection.location_data
    if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
        raise ValueError(
            'LocationData must be relative for this drawing funtion to work.')
    # Draws keypoints.
    # for keypoint in location.relative_keypoints:
    #   keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
    #                                                  image_cols, image_rows)
    #   cv2.circle(image, keypoint_px, keypoint_drawing_spec.circle_radius,
    #              keypoint_drawing_spec.color, keypoint_drawing_spec.thickness)
    # # Draws bounding box if exists.
    if not location.HasField('relative_bounding_box'):
        return
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
        image_rows)
    rect_end_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
        image_rows)
    print(f"rect_start_point : {rect_start_point}")
    print(f"rect_end_point : {rect_end_point}")
    return rect_start_point, rect_end_point
    # cv2.rectangle(image, rect_start_point, rect_end_point,
    #               bbox_drawing_spec.color, bbox_drawing_spec.thickness)


# For webcam input:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
head_text = 'processing head'
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        head_text = head_main_test.main(image.copy())
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_orginal = image.copy()
        if results.detections:
            for detection in results.detections:
                rect_start_point, rect_end_point = get_roi_face(image, detection)
                try:
                    x1, y1 = rect_start_point[0], rect_start_point[1]
                    x2, y2 = rect_end_point[0], rect_end_point[1]
                    img_orginal = img_orginal[y1-100:y2+100, x1-90:x2+90]
                    img_orginal = cv2.resize(img_orginal, (300, 300))
                    results = face_mesh.process(img_orginal.copy())
                    img_h, img_w, img_c = img_orginal.shape
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            for idx, lm in enumerate(face_landmarks.landmark):
                                if idx == 1:
                                    # nose
                                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                                    center = (x, y)
                                    left_eye = [x, y]
                                    thickness = 5
                                    img_orginal = cv2.circle(img_orginal, center, radius, (255,0,0), thickness)

                            if (125 < y < 175 and x < 125) or (y>175 and x<125 and y>=x) or (y<125 and x<125 and y>=x):
                                cv2.putText(img_orginal, "Looking Right", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif (125 < y < 175 and x > 175) or (y>175 and x>175 and y<=x) or (y<125 and x>175 and y<=x):
                                cv2.putText(img_orginal, "Looking Left", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif (125 < x < 175 and y > 175) or (x>175 and y>175 and y>=x) or (x<125 and y>175 and y>=x):
                                cv2.putText(img_orginal, "Looking Down", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif (125 < x < 175 and y<125) or (x>175 and y<125 and y<=x) or (x<125 and y<125 and y<=x):
                                cv2.putText(img_orginal, "Looking Up", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif (125 < y < 175 and 125 < x < 175):
                                cv2.putText(img_orginal, "Forward", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            else:
                                cv2.putText(img_orginal, head_text , (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # img_orginal = cv2.cvtColor(img_orginal, cv2.COLOR_BGR2RGB)
                    # results = face_detection.process(img_orginal)
                    # img_orginal.flags.writeable = True
                    # image = cv2.cvtColor(img_orginal, cv2.COLOR_RGB2BGR)
                    # if results.detections:
                    #   for detection in results.detections:
                    #     mp_drawing.draw_detection(img_orginal, detection)
                    cv2.circle(img_orginal, (150, 150), radius, color, thickness)
                except Exception as e:
                    print(f"error : {e}")
                    pass
                # mp_drawing.draw_detection(image, detection)
        # Flip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Face Detection', cv2.flip(img_orginal, 1))
        cv2.imshow('MediaPipe Face Detection', img_orginal)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
