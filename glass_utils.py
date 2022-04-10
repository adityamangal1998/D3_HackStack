from typing import List, Optional, Tuple, Union
import math

import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


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


def cropped_face(image):
    try:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections is not None:
            for detection in results.detections:
                image_rows, image_cols, _ = image.shape
                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box
                rect_start_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                    image_rows)
                rect_end_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + +relative_bounding_box.height, image_cols,
                    image_rows)
                y1 = rect_start_point[1]
                y2 = rect_end_point[1]
                x1 = rect_start_point[0]
                x2 = rect_end_point[0]
                return image[y1:y2, x1:x2, :],'face'
        return image,'no_face'
    except:
        return image,'no_face'
