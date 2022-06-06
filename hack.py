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
counter_sleep = 0
counter_drowsiness = 0
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
        Eye_val_img = np.zeros([300, 800, 3], dtype=np.uint8)
        Eye_val_img.fill(0)
        Mouth_val_img = np.zeros([300, 800, 3], dtype=np.uint8)
        Mouth_val_img.fill(0)
        Head_val_img = np.zeros([300, 800, 3], dtype=np.uint8)
        Head_val_img.fill(0)
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
                    original_frame = cv2.flip(original_frame, 1)
                    if params['result_eye'] == 'sleeping' or params['result_head'] == 'sleeping':
                        cv2.putText(original_frame, 'sleeping', (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1,
                                    (0, 0, 255), 2)
                        counter_sleep = counter_sleep + 1
                    if params['result_eye'] == "drowsiness" or params['result_mouth'] == "drowsiness":
                        cv2.putText(original_frame, "drowsiness", (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1,
                                    (0, 0, 255), 2)
                        counter_drowsiness = counter_drowsiness + 1
                    if 100 > counter_sleep > 0:
                        cv2.putText(original_frame, 'sleeping', (20, 50),cv2.FONT_HERSHEY_TRIPLEX, 1,
                                    (0, 0, 255), 2)
                        counter_sleep = counter_sleep + 1
                    if 50 > counter_drowsiness > 0:
                        cv2.putText(original_frame, "drowsiness", (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1,
                                    (0, 0, 255), 2)
                        counter_drowsiness = counter_drowsiness + 1
                    if counter_drowsiness > 50:
                        counter_drowsiness = 0
                    if counter_sleep > 100:
                        counter_sleep = 0
                    params['result_eye'] == ''
                    params['result_head'] == ''

        except Exception as e:
            print(f"error : {e}")
            params['head_text'] = 'processing head'
            params['glass_text'] = 'processing glass'
            params['person_text'] = 'processing person'
            params['eye_ratio'] = 'processing eye'
            params['mouth_ratio'] = 'processing mouth'

        # cv2.putText(Eye_val_img, "Head -> " + params['head_text_1'], (40, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(canvas, "Wearing-> " + params['glass_text'], (40, 110), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(canvas, "Eye ratio -> " + str(params['eye_ratio']), (40, 170), cv2.FONT_HERSHEY_TRIPLEX, 1,
        #             (0, 0, 255), 2)
        # cv2.putText(canvas, "Mouth ratio -> " + str(params['mouth_ratio']), (40, 230), cv2.FONT_HERSHEY_TRIPLEX, 1,
        #             (0, 0, 255), 2)

        # $$____$$   Cal_img
        # Eye
        cv2.putText(Eye_val_img, " Blink Trigger Time  -> " + str(params['eye_close_time_stamp']), (40, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        cv2.putText(Eye_val_img, " Eye Blink Time -> " + str(time.time() - params['eye_close_time_stamp']), (40, 90), cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 0, 255), 2)
        cv2.putText(Eye_val_img, " Blink count -> " + str(len(params['eye_blink_stamp'])), (40, 130),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 0, 255), 2)
        # Mouth
        cv2.putText(Mouth_val_img, " Yawn Trigger Time  -> " + str(params['mouth_open_time_stamp']), (40, 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        cv2.putText(Mouth_val_img, " Yawn Time -> " + str(time.time() - params['mouth_open_time_stamp']), (40, 90),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 0, 255), 2)
        cv2.putText(Mouth_val_img, " Yawn count -> " + str(len(params['mouth_yawn_stamp'])), (40, 130),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 0, 255), 2)
        # Head
        cv2.putText(Head_val_img, " Head down Trigger Time  -> " + str(params['head_down_time_stamp']), (40, 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        cv2.putText(Head_val_img, " Head Down Time -> " + str(time.time() - params['head_down_time_stamp']), (40, 90),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 0, 255), 2)
        cv2.putText(Head_val_img, " Head Down count -> " + str(len(params['head_time_stamp'])), (40, 130),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 0, 255), 2)

        # $$___$$    Windows

        # cv2.imshow('canvas', canvas)
        cv2.imshow('head image', head_img)
        cv2.imshow('person', original_frame)
        cv2.imshow('facemesh_tesselation', cv2.flip(facemesh_tesselation, 1))
        cv2.imshow('Head_val_img',Head_val_img)
        cv2.imshow('Eye_val_img', Eye_val_img)
        cv2.imshow('Mouth_val_img', Mouth_val_img)
        # cv2.imshow('facemesh_contours', cv2.flip(facemesh_contours, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
