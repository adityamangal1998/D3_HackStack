# we have to optimise the code by taking face only one time
import cv2

import calculate
# import head_main
# import glass_main
import time
import person_main
import yawn_and_eye
from utils import debug
import core

logger = []
debug_bool = True

start_time_person = int(time.time())
start_time_glass = int(time.time())

logger.append("Looking Forward")
logger.append("Wearing glass")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

params = {
    'glass_text': 'glass',
    'head_text': 'processing head',
    'person_text': 'processing person',
    'eye_ratio': 'processing eye',
    'mouth_ratio': 'processing mouth',
    'eye_close_flag': False,
    'eye_close_time_stamp': 0,
    'eye_time_stamp': [],
    'eye_blink_stamp': [],
    'head_time_stamp': [],
    'ref_epoch': 0,
    'eye_closed_counter': 0,
    'mouth_open_counter': 0,
    'head_down_counter': 0,
    'eye_blink_counter': 0,
    'mouth_yawn_counter': 0,
    'result': ""
}

while cap.isOpened():
    success, image = cap.read()
    if time.time() - start_time_glass > 10:
        params['person_text'] = person_main.main(image.copy())
    # params['person_text'] = "person"
    # if params['person_text'] == "person":
        # params['head_text'], head_x, head_y, image = head_main.main(image.copy())
        # if params['head_text'] != "No Person":
        #     params['mouth_ratio'], params['eye_ratio'] = yawn_and_eye.main(image.copy())
        #     params = core.main(params)
        #     params = calculate.main(params)
        #     if params['result'] == 'sleeping':
        #         debug(f"Sleeping : {params['result']}", debug_bool)
        #         break
        #     if params['result'] == "drowsiness":
        #         debug(f"Drowsiness : {params['result']}", debug_bool)
        #         params['eye_blink_stamp'] = []

    else:
        params['head_text'] = 'processing head'
        params['glass_text'] = 'processing glass'
        params['person_text'] = 'processing person'
        params['eye_ratio'] = 'processing eye'
        params['mouth_ratio'] = 'processing mouth'
    image = cv2.flip(image, 1)
    cv2.putText(image, params['head_text'], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, params['glass_text'], (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, str(params['eye_ratio']), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, str(params['mouth_ratio']), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
# with open('forward_face.txt', 'w') as f:
#     for item in logger:
#         f.write("%s\n" % item)
