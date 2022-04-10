# we have to optimise the code by taking face only one time
import cv2
import head_main
import glass_main
import time
import person_main
import yawn_and_eye
from utils import debug

debug_bool = True
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count_head = 0
count_glass = 0
start_time_head = int(time.time())
start_time_glass = int(time.time())
head_text = 'processing head'
glass_text = 'processing glass'
person_text = 'processing person'
eye_text = 'processing eye'
mouth_text = 'processing mouth'
while cap.isOpened():
    success, image = cap.read()
    if time.time() - start_time_glass > 10:
        person_text = person_main.main(image.copy())
        debug(f"person_text : {person_text}",debug_bool)
    if person_text == "person":
        head_text = head_main.main(image.copy())
        mouth_text, eye_text = yawn_and_eye.main(image.copy())
        if mouth_text == "":
            mouth_text = 'processing mouth'
        if eye_text == "":
            eye_text = 'processing eye'
        debug(f"head_text : {head_text}",debug_bool)
        debug(f"mouth_text : {mouth_text}",debug_bool)
        debug(f"eye_text : {eye_text}",debug_bool)
        if time.time() - start_time_head > 1:
            debug(f"time of head : {time.time() - start_time_head}",debug_bool)
            start_time_head = time.time()
            if "Looking Down" in head_text:
                count_head = count_head + 1
                debug(f"count_head : {count_head}",debug_bool)
            else:
                count_head = 0
        if count_head > 10:
            debug("You are sleeping",debug_bool)
        # after 1 min, we will check for glass
        if time.time() - start_time_glass > 10:
            debug(f"time of glass : {time.time() - start_time_glass}",debug_bool)
            start_time_glass = time.time()
            if head_text != "No Person":
                glass_text = glass_main.main(image.copy())
                if 'no glass' == glass_text:
                    # we can rely on eye and mouth and head
                    pass
                elif 'glass' == glass_text:
                    # we can rely on mouth and head
                    pass
                elif 'Processing' == glass_text:
                    # we can rely on head
                    pass
    else:
        head_text = 'processing head'
        glass_text = 'processing glass'
        person_text = 'processing person'
        eye_text = 'processing eye'
        mouth_text = 'processing mouth'
    image = cv2.flip(image, 1)
    cv2.putText(image, head_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, glass_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, eye_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, mouth_text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
