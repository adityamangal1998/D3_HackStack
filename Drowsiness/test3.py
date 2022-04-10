import cv2
import mediapipe as mp
import math
import numpy as np
from typing import List, Mapping, Optional, Tuple, Union
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
# from mediapipe.framework.formats import landmark_pb2
# import head_main_test_1


def display():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    radius = 2
    color = (0, 255, 0)
    thickness = -1
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # face_image = np.ones((300, 300, 3)) * 255
            face_image = np.zeros([300, 300, 3], dtype=np.uint8)
            face_image.fill(255)
            face_image_1 = np.zeros([300, 300, 3], dtype=np.uint8)
            face_image_1.fill(255)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_orginal = image.copy()
            if results.detections:
                for detection in results.detections:
                    rect_start_point, rect_end_point = get_roi_face(
                        image, detection)
                    # try:
                    x1, y1 = rect_start_point[0], rect_start_point[1]
                    x2, y2 = rect_end_point[0], rect_end_point[1]
                    img_orginal = img_orginal[y1 - 70:y2 + 30, x1 - 30:x2 + 30]
                    img_orginal = cv2.resize(img_orginal, (300, 300))
                    face_image = np.zeros([300, 300, 3], dtype=np.uint8)
                    face_image.fill(255)
                    face_image_1 = np.zeros([300, 300, 3], dtype=np.uint8)
                    face_image_1.fill(255)
                    results = face_mesh.process(img_orginal.copy())
                    img_h, img_w, img_c = img_orginal.shape
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=face_image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_contours_style())
                            mp_drawing.draw_landmarks(
                                image=face_image_1,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_tesselation_style())
                            for idx, lm in enumerate(face_landmarks.landmark):
                                if idx == 1:
                                    # nose
                                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                                    center = (x, y)
                                    left_eye = [x, y]
                                    thickness = 5
                                    img_orginal = cv2.circle(
                                        img_orginal, center, radius, (255, 0, 0), thickness)
                    cv2.circle(img_orginal, (150, 150),
                               radius, color, thickness)
                    # except Exception as e:
                    #     print(f"error : {e}")
                    #     pass
                    # mp_drawing.draw_detection(image, detection)
            # Flip the image horizontally for a selfie-view display.

            # cv2.imshow('MediaPipe Face Detection', cv2.flip(img_orginal, 1))
            # cv2.imshow('Original Image', cv2.flip(image, 1))
            # cv2.imshow('Face Detection', face_image)
            # cv2.imshow('Face Detection 1', cv2.flip(face_image_1, 1))
                face = cv2.flip(img_orginal, 1)
                original = cv2.flip(image, 1)
                face = cv2.flip(img_orginal, 1)
                mesh = cv2.flip(face_image_1, 1)
            cv2.imshow('MediaPipe Face Detection', face)
            cv2.imshow('Original Image', original)
            cv2.imshow('Outline', face_image)
            cv2.imshow('Mesh', mesh)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


display()
