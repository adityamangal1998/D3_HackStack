import time
import cv2
import mediapipe as mp
from utils import debug
import calculate
import core
import numpy as np
import glass_main
import head_main
import utils
import head_hack
import yawn_and_eye

params = {
    'glass_text': 'glass',
    'head_text_1': 'processing head',
    'head_text_2': 'processing head',
    'person_text': 'processing person',
    'eye_ratio': 'processing eye',
    'mouth_ratio': 'processing mouth',
    'eye_close_flag': False,
    'eye_close_time_stamp': 0,
    'eye_time_stamp': [],
    'mouth_time_stamp': [],
    'mouth_open_time_stamp': 0,
    'head_down_time_stamp': 0,
    'mouth_open_flag': False,
    'head_down_flag': False,
    'eye_blink_stamp': [],
    'mouth_yawn_stamp': [],
    'head_time_stamp': [],
    'ref_epoch': 0,
    'result_eye': "",
    'result_mouth': "",
    'result_head': ""
}
debug_bool = True
radius = 2
color = (0, 255, 0)
thickness = -1
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
start_time_glass = int(time.time())
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        success, image = cap.read()
        original_frame = image.copy()
        cropped_face = np.zeros([300, 300, 3], dtype=np.uint8)
        cropped_face.fill(255)
        facemesh_tesselation = np.zeros([300, 300, 3], dtype=np.uint8)
        facemesh_tesselation.fill(255)
        facemesh_contours = np.zeros([300, 300, 3], dtype=np.uint8)
        facemesh_contours.fill(255)
        head_img = np.zeros([300, 300, 3], dtype=np.uint8)
        head_img.fill(255)
        canvas = np.zeros([250, 800, 3], dtype=np.uint8)
        canvas.fill(0)
        try:
            results = face_detection.process(image)
            img_original = image.copy()
            if results.detections:
                for detection in results.detections:
                    rect_start_point, rect_end_point = utils.get_roi_face(image, detection)
                    x1, y1 = rect_start_point[0], rect_start_point[1]
                    x2, y2 = rect_end_point[0], rect_end_point[1]
                    img_original = img_original[y1 - 100:y2 + 100, x1 - 90:x2 + 90]
                    img_original = cv2.resize(img_original, (300, 300))
                    if time.time() - start_time_glass > 10:
                        start_time_glass = time.time()
                        glass_text = glass_main.main(img_original.copy())
                        params['glass_text'] = glass_text
                    params['head_text_1'], head_img, facemesh_tesselation, facemesh_contours = head_hack.main(
                        img_original.copy())
                    params['head_text_2'], head_x, head_y = head_main.main(original_frame.copy())
                    if params['head_text_1'] == "processing head" and params['head_text_2'] == "Looking Up":
                        params['head_text_1'] = "Looking Up"
                    params['mouth_ratio'], params['eye_ratio'] = yawn_and_eye.main(image.copy())
                    params = core.eye_main(params)
                    params = calculate.eye_calculate_main(params)
                    params = core.mouth_main(params)
                    params = calculate.mouth_calculate_main(params)
                    params = core.head_main(params)
                    params = calculate.head_calculate_main(params)
                    if params['result_eye'] == 'sleeping':
                        debug(f"Sleeping : {params['result_eye']}", debug_bool)
                        break
                    if params['result_head'] == 'sleeping':
                        debug(f"Sleeping HEAD : {params['result_head']}", debug_bool)
                        break
                    if params['result_eye'] == "drowsiness":
                        debug(f"Drowsiness EYE: {params['result_eye']}", debug_bool)
                        params['eye_blink_stamp'] = []
                    if params['result_mouth'] == "drowsiness":
                        debug(f"Drowsiness MOUTH : {params['result_mouth']}", debug_bool)
                        params['mouth_yawn_stamp'] = []
        except Exception as e:
            print(f"error : {e}")
            params['head_text'] = 'processing head'
            params['glass_text'] = 'processing glass'
            params['person_text'] = 'processing person'
            params['eye_ratio'] = 'processing eye'
            params['mouth_ratio'] = 'processing mouth'
        image = cv2.flip(image, 1)

        cv2.putText(canvas, "head -> " + params['head_text_1'], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(canvas, "wearing-> " + params['glass_text'], (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(canvas, "eye ratio -> " + str(params['eye_ratio']), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.putText(canvas, "mouth ratio -> " + str(params['mouth_ratio']), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.imshow('canvas', canvas)
        cv2.imshow('head image', head_img)
        cv2.imshow('person', cv2.flip(original_frame, 1))
        cv2.imshow('facemesh_tesselation', cv2.flip(facemesh_tesselation, 1))
        cv2.imshow('facemesh_contours', cv2.flip(facemesh_contours, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
