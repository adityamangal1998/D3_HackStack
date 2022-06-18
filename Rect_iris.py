from operator import rshift
import cv2 as cv
import os
import mediapipe as mp
import logging
import time
from typing import Tuple, Union
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
import math
import numpy as np
import iris_seg
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
def normalized_to_pixel_coordinates(
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
    if not location.HasField('relative_bounding_box'):
        return
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
        image_rows)
    rect_end_point = normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
        image_rows)
    return rect_start_point, rect_end_point

model_sr = r"D:\teco_nico\D3_HackStack-main\D3_HackStack-main\ESPCN_x4.pb"
modelName = model_sr.split(os.path.sep)[-1].split("_")[0].lower()
modelScale = model_sr.split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])
sr = cv.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_sr)
sr.setModel(modelName, modelScale)

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# LEFT_IRIS = [474, 475, 476, 477]
# RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474]
RIGHT_IRIS = [469]
Blink_count =0
cap = cv.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        try:
            results = face_detection.process(frame)
            result = face_mesh.process(rgb_frame)
            img_original = frame.copy()
            if results.detections:
                # print(results.detections)
                for detection in results.detections:
                    rect_start_point, rect_end_point = get_roi_face(frame, detection)
                    x1, y1 = rect_start_point[0], rect_start_point[1]
                    x2, y2 = rect_end_point[0], rect_end_point[1]
                    img_original = img_original[y1 - 100:y2 + 100, x1 - 90:x2 + 90]
                    img_original = cv.resize(img_original, (500,500))
                box,contour_count,contour_frame,Area,width,height = iris_seg.main(img_original)
                new = np.zeros(img_original.shape[:2], np.uint8)
                new = cv.drawContours(new,[box], -1, (255, 0, 0), 1)
                print(contour_count)
                print(Area)
                print(height)

                if height <= 8:
                    Blink_count += 1
                    cv.putText(contour_frame,"__Blink__",(50,50),cv.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)
                cv.putText(contour_frame,str(Blink_count), (50, 100), cv.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"error : {e}")
        cv.imshow('img_original', img_original)
        cv.imshow("new", new)
        cv.imshow('res', contour_frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
